import streamlit as st
import asyncio
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, END, START
from typing import List, Dict, Any, Annotated, TypedDict
import pandas as pd
import json
import logging
from datetime import datetime
import warnings
import operator
import torch
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    """State management for LangGraph with proper type definitions and reducers"""
    original_query: str
    enhanced_queries: Annotated[List[str], operator.add]  # Use reducer for list concatenation
    retrieved_docs: Annotated[List[Dict], operator.add]  # Use reducer for list concatenation
    response: str
    embeddings: Any
    use_full_kb: bool
    processing_mode: str

class ElasticsearchConnector:
    """Handle Elasticsearch connections and data retrieval"""
    
    def __init__(self, host="localhost", port=9200):
        try:
            self.es = Elasticsearch([f"http://{host}:{port}"])
            # Test connection
            self.es.info()
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
            
        self.system_indexes = [
            '.kibana', '.security', '.monitoring', '.watcher', 
            '.ml-', '.transform', '.async-search', '.fleet'
        ]
    
    def get_data_indexes(self):
        """Get all non-system indexes"""
        try:
            all_indexes = self.es.indices.get_alias(index="*")
            data_indexes = []
            
            for index_name in all_indexes.keys():
                if not any(sys_idx in index_name for sys_idx in self.system_indexes):
                    if not index_name.startswith('.'):
                        data_indexes.append(index_name)
            
            return data_indexes
        except Exception as e:
            logger.error(f"Error getting indexes: {e}")
            return []
    
    def extract_documents_from_index(self, index_name, size=1000):
        """Extract all documents from a specific index as individual documents"""
        try:
            query = {"query": {"match_all": {}}}
            response = self.es.search(index=index_name, body=query, size=size, scroll='2m')
            
            documents = []
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            
            while hits:
                for hit in hits:
                    doc_content = self._extract_text_from_source(hit['_source'])
                    if doc_content.strip():
                        # Store each document as a separate entity
                        documents.append({
                            'content': doc_content,
                            'metadata': {
                                'index': index_name,
                                'id': hit['_id'],
                                'source': hit['_source'],
                                'doc_type': 'full_document'  # Mark as complete document
                            }
                        })
                
                # Get next batch
                try:
                    response = self.es.scroll(scroll_id=scroll_id, scroll='2m')
                    hits = response['hits']['hits']
                except:
                    break
            
            return documents
        except Exception as e:
            logger.error(f"Error extracting documents from {index_name}: {e}")
            return []
    
    def _extract_text_from_source(self, source):
        """Extract text content from document source"""
        text_parts = []
        
        def extract_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (str, int, float)):
                        if isinstance(value, str) and len(value.strip()) > 0:
                            text_parts.append(f"{prefix}{key}: {value}")
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, f"{prefix}{key}.")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{prefix}[{i}].")
        
        extract_recursive(source)
        return " | ".join(text_parts)

class EmbeddingManager:
    """Handle document embeddings using SentenceTransformers and Elasticsearch - Store each document separately"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', es_host="localhost", es_port=9200):
        try:
            # Initialize model with explicit device management
            self.model = SentenceTransformer(model_name, device='cpu')
            # Force model to load weights
            self.model.eval()
            
            self.es = Elasticsearch([f"http://{es_host}:{es_port}"])
            self.embedding_index = "document_embeddings_individual"
            self._ensure_embedding_index()
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingManager: {e}")
            raise
    
    def _ensure_embedding_index(self):
        """Create the embedding index if it doesn't exist"""
        # Delete index if it exists
        if self.es.indices.exists(index=self.embedding_index):
            self.es.indices.delete(index=self.embedding_index)
            logger.info(f"Deleted existing index: {self.embedding_index}")
        
        # Create index with vector mapping
        mapping = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "metadata": {"type": "object"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,  # Dimension for all-MiniLM-L6-v2
                        "index": True,
                        "similarity": "cosine"
                    },
                    "doc_length": {"type": "integer"},
                    "source_index": {"type": "keyword"}
                }
            }
        }
        self.es.indices.create(index=self.embedding_index, body=mapping)
        logger.info(f"Created new index: {self.embedding_index}")
    
    def create_embeddings(self, documents: List[Dict]):
        """Create embeddings for individual documents and store in Elasticsearch"""
        try:
            if not documents:
                logger.warning("No documents provided for embedding creation")
                return False
            
            # Filter out very short documents that might not be meaningful
            valid_docs = []
            for doc in documents:
                if len(doc['content'].strip()) > 50:  # Only keep documents with substantial content
                    valid_docs.append(doc)
            
            if not valid_docs:
                logger.warning("No valid documents after filtering")
                return False
            
            logger.info(f"Processing {len(valid_docs)} individual documents for embedding creation")
            
            # Create and store embeddings in batches
            batch_size = 50  # Smaller batch size for individual documents
            total_stored = 0
            
            for i in range(0, len(valid_docs), batch_size):
                batch = valid_docs[i:i + batch_size]
                contents = [doc['content'] for doc in batch]
                
                # Create embeddings for the batch with explicit device management
                with torch.no_grad():
                    embeddings = self.model.encode(contents, convert_to_tensor=False, show_progress_bar=True)
                
                # Prepare bulk insert
                bulk_data = []
                for doc, embedding in zip(batch, embeddings):
                    bulk_data.append({
                        "index": {
                            "_index": self.embedding_index
                        }
                    })
                    bulk_data.append({
                        "content": doc['content'],
                        "metadata": doc['metadata'],
                        "embedding": embedding.tolist(),
                        "doc_length": len(doc['content']),
                        "source_index": doc['metadata']['index']
                    })
                
                # Bulk insert to Elasticsearch
                if bulk_data:
                    response = self.es.bulk(body=bulk_data)
                    # Check for errors
                    if not response.get('errors', False):
                        total_stored += len(batch)
                    else:
                        logger.warning(f"Some documents in batch {i//batch_size + 1} failed to store")
            
            logger.info(f"Successfully stored embeddings for {total_stored} individual documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            return False
    
    def search_similar_documents(self, query: str, top_k=50, min_score=0.0):
        """Search for similar documents using vector similarity in Elasticsearch
        Args:
            query: Search query
            top_k: Number of documents to retrieve (default 50 for full KB mode)
            min_score: Minimum similarity score threshold
        """
        try:
            # Create query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Prepare vector search query
            search_query = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding.tolist()}
                        },
                        "min_score": min_score + 1.0  # Adjust for the +1.0 in the script
                    }
                },
                "size": top_k,
                "_source": ["content", "metadata", "doc_length", "source_index"]
            }
            
            # Execute search
            response = self.es.search(
                index=self.embedding_index,
                body=search_query
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                similarity_score = hit['_score'] - 1.0  # Adjust score to be between 0 and 1
                results.append({
                    'document': {
                        'content': hit['_source']['content'],
                        'metadata': hit['_source']['metadata']
                    },
                    'score': similarity_score,
                    'doc_length': hit['_source'].get('doc_length', 0),
                    'source_index': hit['_source'].get('source_index', 'unknown')
                })
            
            logger.info(f"Retrieved {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_top_k_documents(self, query: str, k=5):
        """Get top K most similar documents for smart mode"""
        return self.search_similar_documents(query, top_k=k, min_score=0.1)
    
    def get_all_similar_documents(self, query: str, min_similarity=0.05):
        """Get all similar documents above threshold for full KB mode"""
        return self.search_similar_documents(query, top_k=100, min_score=min_similarity)

class PromptEngineer:
    """Handle prompt engineering and enhancement"""
    
    @staticmethod
    def enhance_prompts(original_query: str) -> List[str]:
        """Create enhanced versions of the same prompt for better retrieval"""
        enhanced_queries = [
            original_query,  # Original
            f"Please provide detailed information about: {original_query}",  # Detailed
            f"What are the key aspects and details regarding {original_query}?",  # Questioning
            f"Explain {original_query} with relevant context and examples",  # Contextual
        ]
        return enhanced_queries

class OllamaLLM:
    """Handle Ollama LLM interactions with improved knowledge base processing"""
    
    def __init__(self, model_name='llama3.2'):
        self.model_name = model_name
        try:
            self.client = ollama.Client()
            # Test connection
            self.client.list()
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def generate_response(self, query: str, context_docs: List[Dict], mode='smart') -> str:
        """Generate response using context documents
        Args:
            query: User query
            context_docs: Retrieved documents with similarity scores
            mode: 'smart' (top 5) or 'full_kb' (all similar documents)
        """
        try:
            if mode == 'full_kb':
                return self._generate_with_full_kb(query, context_docs)
            else:
                return self._generate_with_smart_mode(query, context_docs)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating the response."
    
    def _generate_with_smart_mode(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using top 5 most relevant documents"""
        # Take only top 5 documents
        top_docs = context_docs[:5]
        context = self._prepare_context(top_docs, mode='smart')
        
        prompt = f"""
Based on the following top 5 most relevant documents, please answer the user's question. 
Use the information from these documents to provide a focused and accurate response.

Relevant Documents:
{context}

Question: {query}

Please provide a clear and concise answer based on the most relevant information:
"""
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': 2000,
                    'num_ctx': 16384  # Increased context window
                }
            )
            return response['response']
        except Exception as e:
            logger.error(f"Error in smart mode generation: {e}")
            return "Sorry, I couldn't generate a response at this time."
    
    def _generate_with_full_kb(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using all similar documents as knowledge base"""
        # Use all provided documents as comprehensive knowledge base
        full_context = self._prepare_context(context_docs, mode='full_kb')
        
        prompt = f"""
You have access to a comprehensive knowledge base containing all documents similar to the user's query.
Please provide a detailed and thorough answer using any relevant information from this knowledge base.
Draw insights from multiple documents to give a comprehensive response.

Complete Knowledge Base (All Similar Documents):
{full_context}

Question: {query}

Please provide a comprehensive answer drawing from all relevant information in the knowledge base:
"""
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'num_predict': 4000,  # Increased for longer responses
                    'num_ctx': 32768,  # Maximum context window
                    'repeat_penalty': 1.1,  # Slightly increased to reduce repetition
                    'top_k': 40  # Increased for better diversity
                }
            )
            return response['response']
        except Exception as e:
            logger.error(f"Error in full KB generation: {e}")
            return "Sorry, I couldn't generate a comprehensive response at this time."
    
    def _prepare_context(self, docs: List[Dict], mode='smart') -> str:
        """Prepare context from documents based on mode"""
        if not docs:
            return "No relevant documents found."
        
        context_parts = []
        
        if mode == 'smart':
            # For smart mode, show top 5 with detailed information
            for i, doc_data in enumerate(docs[:5], 1):
                doc = doc_data['document']
                score = doc_data['score']
                source = doc_data.get('source_index', 'unknown')
                
                context_parts.append(
                    f"Document {i} (Relevance: {score:.3f}, Source: {source}):\n"
                    f"{doc['content']}\n"
                )
        else:
            # For full KB mode, organize by source and show similarity scores
            docs_by_source = {}
            for doc_data in docs:
                source = doc_data.get('source_index', 'unknown')
                if source not in docs_by_source:
                    docs_by_source[source] = []
                docs_by_source[source].append(doc_data)
            
            for source, source_docs in docs_by_source.items():
                context_parts.append(f"=== Source: {source} ===")
                for i, doc_data in enumerate(source_docs, 1):
                    doc = doc_data['document']
                    score = doc_data['score']
                    
                    # Limit document length for context window management
                    content = doc['content']
                    if len(content) > 1500:  # Increased from 800 to 1500
                        content = content[:1500] + "..."
                    
                    context_parts.append(
                        f"Doc {i} (Similarity: {score:.3f}):\n{content}\n"
                    )
                context_parts.append("")
        
        return "\n---\n".join(context_parts)

class RAGWorkflow:
    """LangGraph workflow for RAG process with individual document storage"""
    
    def __init__(self, es_connector, embedding_manager, llm):
        self.es_connector = es_connector
        self.embedding_manager = embedding_manager
        self.llm = llm
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """Create LangGraph workflow with proper state management"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("enhance_query", self._enhance_query_node)
        workflow.add_node("retrieve_smart_docs", self._retrieve_smart_docs_node)
        workflow.add_node("retrieve_full_kb_docs", self._retrieve_full_kb_docs_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Add conditional edges based on processing mode
        def route_processing(state: RAGState) -> str:
            """Route to appropriate retrieval node based on mode"""
            if state.get('use_full_kb', False):
                return "retrieve_full_kb_docs"
            else:
                return "retrieve_smart_docs"
        
        # Set up the workflow
        workflow.add_edge(START, "enhance_query")
        workflow.add_conditional_edges(
            "enhance_query",
            route_processing,
            {
                "retrieve_smart_docs": "retrieve_smart_docs",
                "retrieve_full_kb_docs": "retrieve_full_kb_docs"
            }
        )
        workflow.add_edge("retrieve_smart_docs", "generate_response")
        workflow.add_edge("retrieve_full_kb_docs", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _enhance_query_node(self, state: RAGState) -> Dict[str, Any]:
        """Enhance the original query"""
        enhanced_queries = PromptEngineer.enhance_prompts(state['original_query'])
        return {'enhanced_queries': enhanced_queries}
    
    def _retrieve_smart_docs_node(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve top 5 most similar documents for smart mode"""
        # Use the original query for similarity search
        query = state['original_query']
        results = self.embedding_manager.get_top_k_documents(query, k=5)
        
        logger.info(f"Smart mode: Retrieved {len(results)} documents")
        return {'retrieved_docs': results}
    
    def _retrieve_full_kb_docs_node(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve all similar documents for full KB mode"""
        # Use the original query for similarity search
        query = state['original_query']
        results = self.embedding_manager.get_all_similar_documents(query, min_similarity=0.05)
        
        logger.info(f"Full KB mode: Retrieved {len(results)} documents")
        return {'retrieved_docs': results}
    
    def _generate_response_node(self, state: RAGState) -> Dict[str, Any]:
        """Generate response using retrieved documents"""
        mode = 'full_kb' if state.get('use_full_kb', False) else 'smart'
        retrieved_docs = state.get('retrieved_docs', [])
        
        response = self.llm.generate_response(
            state['original_query'], 
            retrieved_docs, 
            mode=mode
        )
        
        return {'response': response}
    
    def process_query(self, query: str, mode='smart') -> Dict:
        """Process a query through the RAG workflow
        
        Args:
            query: User query
            mode: 'smart' (top 5 docs) or 'full_kb' (all similar docs)
        """
        # Initialize state properly
        initial_state: RAGState = {
            'original_query': query,
            'enhanced_queries': [],
            'retrieved_docs': [],
            'response': "",
            'embeddings': None,
            'use_full_kb': mode == 'full_kb',
            'processing_mode': mode
        }
        
        try:
            # Run through the workflow
            result = self.workflow.invoke(initial_state)
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'original_query': query,
                'response': f"Error processing query: {str(e)}",
                'retrieved_docs': [],
                'enhanced_queries': []
            }

# Streamlit App
def main():
    st.set_page_config(
        page_title="Enhanced RAG System - Individual Document Storage",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Enhanced RAG System - Individual Document Storage")
    st.markdown("Each document stored separately with cosine similarity vector search")
    st.markdown("---")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'initialization_complete' not in st.session_state:
        st.session_state.initialization_complete = False
    if 'total_documents' not in st.session_state:
        st.session_state.total_documents = 0
    
    # System status sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        if st.session_state.rag_system:
            st.success("✅ RAG System: Ready")
            st.info(f"📊 Total Documents: {st.session_state.total_documents}")
        else:
            st.warning("⚠️ RAG System: Initializing...")
        
        st.markdown("### Mode Explanation")
        st.markdown("**🎯 Smart Mode:** Uses top 5 most similar documents")
        st.markdown("**📚 Full KB Mode:** Uses all similar documents as knowledge base")
        
        if st.button("🔄 Reinitialize System"):
            st.session_state.rag_system = None
            st.session_state.initialization_complete = False
            st.session_state.total_documents = 0
            st.rerun()
    
    # Auto-initialize system on first load
    if st.session_state.rag_system is None and not st.session_state.initialization_complete:
        with st.spinner("Initializing Enhanced RAG system..."):
            try:
                # Initialize components with default settings
                st.info("Connecting to Elasticsearch...")
                es_connector = ElasticsearchConnector("localhost", 9200)
                
                st.info("Loading embedding model...")
                embedding_manager = EmbeddingManager("all-MiniLM-L6-v2")
                
                st.info("Connecting to Ollama...")
                llm = OllamaLLM("llama3.2")
                
                # Get data from Elasticsearch
                st.info("Fetching data indexes...")
                data_indexes = es_connector.get_data_indexes()
                
                if not data_indexes:
                    st.error("❌ No data indexes found in Elasticsearch!")
                    st.info("Make sure Elasticsearch is running and contains data indexes.")
                    st.session_state.initialization_complete = True
                    st.stop()
                
                st.info(f"Found {len(data_indexes)} data indexes: {', '.join(data_indexes)}")
                
                # Extract documents (each document stored individually)
                st.info("Extracting individual documents...")
                all_documents = []
                progress_bar = st.progress(0)
                
                for i, index in enumerate(data_indexes):
                    progress_bar.progress((i + 1) / len(data_indexes))
                    docs = es_connector.extract_documents_from_index(index)
                    all_documents.extend(docs)
                    st.info(f"Extracted {len(docs)} individual documents from {index}")
                
                if not all_documents:
                    st.error("❌ No documents found in any index!")
                    st.session_state.initialization_complete = True
                    st.stop()
                
                st.session_state.total_documents = len(all_documents)
                st.info(f"Total individual documents extracted: {len(all_documents)}")
                
                # Create embeddings for individual documents
                st.info("Creating embeddings for individual documents...")
                if embedding_manager.create_embeddings(all_documents):
                    # Initialize workflow
                    workflow = RAGWorkflow(es_connector, embedding_manager, llm)
                    
                    st.session_state.rag_system = workflow
                    st.session_state.initialization_complete = True
                    st.success("🎉 Enhanced RAG System initialized successfully!")
                    st.success(f"📊 {st.session_state.total_documents} individual documents ready for search")
                    st.rerun()
                else:
                    st.error("❌ Failed to create embeddings!")
                    st.session_state.initialization_complete = True
                    st.stop()
                    
            except Exception as e:
                st.error(f"❌ Error initializing RAG system: {e}")
                st.session_state.initialization_complete = True
                st.info("Please check your Elasticsearch and Ollama connections.")
                st.stop()
    
    # Main chat interface
    if st.session_state.rag_system:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for i, (query, response, docs, mode) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**You [{mode.upper()}]:** {query}")
                st.markdown(f"**Assistant:** {response}")
                
                # Show retrieved documents in expander
                if docs:
                    with st.expander(f"📄 Retrieved Documents ({len(docs)} docs)"):
                        for j, doc_data in enumerate(docs, 1):
                            doc = doc_data['document']
                            score = doc_data['score']
                            source = doc_data.get('source_index', 'unknown')
                            doc_length = doc_data.get('doc_length', 0)
                            
                            st.markdown(f"**Document {j}** - Source: {source}")
                            st.markdown(f"Similarity: {score:.3f} | Length: {doc_length} chars")
                            content_preview = doc['content'][:400] + "..." if len(doc['content']) > 400 else doc['content']
                            st.text(content_preview)
                            st.markdown("---")
                st.markdown("---")
        
        # Query input with mode selection
        with st.form("query_form"):
            user_query = st.text_area("Enter your question:", height=100, 
                                    placeholder="Ask anything about your data...")
            
            # Query processing mode selection
            mode = st.selectbox(
                "Response Mode:",
                options=["smart", "full_kb"],
                format_func=lambda x: {
                    "smart": "🎯 Smart RAG (Top 5 most similar documents)",
                    "full_kb": "📚 Full Knowledge Base (All similar documents)"
                }[x],
                help="Choose how to process your query:\n"
                     "• Smart: Uses top 5 most similar documents for focused answers\n"
                     "• Full KB: Uses all similar documents for comprehensive answers"
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                submit_button = st.form_submit_button("🔍 Search", use_container_width=True, type="primary")
            
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if submit_button and user_query.strip():
            with st.spinner(f"Processing your query using {mode.upper()} mode..."):
                try:
                    # Process query through RAG workflow with selected mode
                    result = st.session_state.rag_system.process_query(user_query, mode=mode)
                    
                    # Add to chat history with mode information
                    st.session_state.chat_history.append((
                        user_query,
                        result['response'],
                        result.get('retrieved_docs', []),
                        mode
                    ))
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
    
    elif st.session_state.initialization_complete:
        st.warning("⚠️ RAG system failed to initialize. Please check your connections and try reinitializing.")
    else:
        st.info("🔄 Enhanced RAG system is initializing... Please wait.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Enhanced RAG System** | Individual Document Storage | "
        "Built with Elasticsearch Dense Vectors, OLLAMA, LangChain, LangGraph, and Streamlit"
    )

if __name__ == "__main__":
    main()
