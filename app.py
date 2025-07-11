import streamlit as st
from elasticsearch import Elasticsearch
import ollama
import json
import pandas as pd
from typing import Dict, List, Any, Tuple, TypedDict
import logging
import re
import os
from datetime import datetime
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the LangGraph agent"""
    user_query: str
    intent: str
    catalog_data: Dict[str, List[str]]
    elasticsearch_query: Dict
    selected_index: str
    raw_data: List[Dict]
    summarized_data: str
    final_response: str
    error: str
    elasticsearch_response: Dict


class SmartBankingAgent:
    """Main agent class with LangGraph workflow"""

    def __init__(self, groq_api_key: str, catalog_index: str = "catalog_index"):
        self.groq_api_key = groq_api_key
        self.catalog_index = catalog_index

        # Initialize components
        self.es = self._init_elasticsearch()
        self.groq_llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.1,
            groq_api_key=groq_api_key
        )
        self.ollama_llm = Ollama(model="mistral:7b-instruct")

        # Build the graph
        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    def _init_elasticsearch(self) -> Elasticsearch:
        """Initialize Elasticsearch connection"""
        try:
            es = Elasticsearch([f"http://localhost:9200"])
            es.info()
            logger.info("Connected to Elasticsearch")
            return es
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("load_catalog", self.load_catalog)
        workflow.add_node("intent_detect", self.intent_detect)
        workflow.add_node("query_generation", self.query_generation)
        workflow.add_node("fetch_data", self.fetch_data)
        workflow.add_node("respond", self.respond)

        # Add edges
        workflow.add_edge("load_catalog", "intent_detect")
        workflow.add_edge("intent_detect", "query_generation")
        workflow.add_edge("query_generation", "fetch_data")
        workflow.add_edge("fetch_data", "respond")
        workflow.add_edge("respond", END)

        # Set entry point
        workflow.set_entry_point("load_catalog")

        return workflow

    def load_catalog(self, state: AgentState) -> AgentState:
        """Load catalog data from Elasticsearch"""
        try:
            logger.info("Loading catalog data...")

            # Get catalog structure
            catalog_query = {
                "query": {"match_all": {}},
                "size": 1000
            }

            # Use new Elasticsearch API (no 'body')
            response = self.es.search(
                index=self.catalog_index,
                query=catalog_query["query"],
                size=catalog_query["size"]
            )

            catalog_data = {}
            for hit in response['hits']['hits']:
                source = hit['_source']
                index_name = source.get('index_name', '')
                columns = source.get('fields', [])  # Updated to 'fields'
                if index_name:
                    catalog_data[index_name] = columns

            state['catalog_data'] = catalog_data
            logger.info(f"Loaded catalog for {len(catalog_data)} indexes")

        except Exception as e:
            logger.error(f"Error loading catalog: {e}")
            state['error'] = f"Error loading catalog: {str(e)}"

        return state

    def intent_detect(self, state: AgentState) -> AgentState:
        """Detect user intent using Groq"""
        try:
            logger.info("Detecting user intent...")

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a banking data analyst. Classify user intent accurately."),
                ("human", """
Analyze the following banking query and determine the user's intent:

Query: "{user_query}"

Intent Types:
1. TABLE - User wants to see raw data in tabular format 
   Examples: "show me all transactions", "display customer data", "list all accounts", "get loan details"

2. CHAT - User wants analysis, explanation, summary, or conversational response
   Examples: "analyze customer behavior", "explain loan trends", "summarize account performance", "what insights can you provide"

Respond with ONLY one word: either "table" or "chat"

Intent:""")
            ])

            response = self.groq_llm.invoke(
                prompt.format_messages(user_query=state['user_query'])
            )

            intent_raw = response.content.strip().lower()
            logger.info(f"LLM intent response: {intent_raw}")

            if intent_raw in ['table', 'chat']:
                state['intent'] = intent_raw
            else:
                # Try to extract intent from response
                if 'table' in intent_raw:
                    state['intent'] = 'table'
                elif 'chat' in intent_raw:
                    state['intent'] = 'chat'
                else:
                    logger.warning(f"Unclear intent response: {intent_raw}, defaulting to 'table'")
                    state['intent'] = 'table'

            logger.info(f"Detected intent: {state['intent']}")

        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            state['error'] = f"Error detecting intent: {str(e)}"
            state['intent'] = 'table'  # Default fallback

        return state

    def query_generation(self, state: AgentState) -> AgentState:
        """Generate Elasticsearch query using Groq"""
        try:
            logger.info("Generating Elasticsearch query...")

            # Format catalog data for prompt
            catalog_info = self._format_catalog_data(state['catalog_data'])
            print(f"\n\n===== CATALOG INFO SENT TO LLM =====\n{catalog_info}\n==============================\n\n")

            # Simplified and cleaner prompt
            prompt_template = """
You are an Elasticsearch expert specializing in financial and banking data analysis.

USER REQUEST: "{user_query}"
USER INTENT: "{intent}"

AVAILABLE INDEXES & STRUCTURE:
{catalog_info}

INSTRUCTIONS:
1. Analyze the user's request and choose the most appropriate index
2. Generate a valid Elasticsearch query in JSON format
3. Use appropriate query types: match, term, range, bool, multi_match, exists, aggregations
4. For aggregations, use sum, avg, value_count, cardinality, min, max as needed
5. Handle date fields with range queries
6. Use _source to limit returned fields when appropriate
7. For text fields, use .keyword for exact matches
8. Only use fields listed in the AVAILABLE INDEXES & STRUCTURE section above. Do NOT use _id or any field not present in the catalog.

RESPONSE FORMAT - Follow this EXACT format:
SELECTED_INDEX: [index_name]
ELASTICSEARCH_QUERY:
{{
  "query": {{
    "match_all": {{}}
  }}
}}

Examples:

For "show me all accounts":
SELECTED_INDEX: total_exposures_index
ELASTICSEARCH_QUERY:
{{
  "query": {{
    "match_all": {{}}
  }},
}}

For "accounts with rating AAA":
SELECTED_INDEX: XYZ_INDEX
ELASTICSEARCH_QUERY:
{{
  "query": {{
    "term": {{
      "rating.keyword": "AAA"
    }}
  }}
}}

For "total exposure by product":
SELECTED_INDEX: ABC_INDEX
ELASTICSEARCH_QUERY:
{{
  "size": 0,
  "aggs": {{
    "by_product": {{
      "terms": {{
        "field": "product.keyword"
      }},
      "aggs": {{
        "total_exposure": {{
          "sum": {{
            "field": "exposure_amt"
          }}
        }}
      }}
    }}
  }}
}}

Now generate the query for the user request:
"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an Elasticsearch expert. Follow the format exactly."),
                ("human", prompt_template)
            ])

            response = self.groq_llm.invoke(
                prompt.format_messages(
                    user_query=state['user_query'],
                    intent=state['intent'],
                    catalog_info=catalog_info
                )
            )

            response_text = response.content.strip()
            logger.info(f"LLM query response: {response_text}")
            print(f"\n\n===== LLM QUERY RESPONSE =====\n{response_text}\n==============================\n\n")

            # Improved parsing logic
            selected_index = None
            query_json = None

            # Extract selected index
            lines = response_text.split('\n')
            for line in lines:
                if line.strip().startswith('SELECTED_INDEX:'):
                    # Extract index name, handling brackets or plain text
                    index_part = line.split('SELECTED_INDEX:')[1].strip()
                    selected_index = index_part.strip('[]').strip()
                    break

            # Extract JSON query - look for the JSON block after "ELASTICSEARCH_QUERY:"
            try:
                # Find the start of the JSON
                json_start = response_text.find('ELASTICSEARCH_QUERY:')
                if json_start != -1:
                    json_part = response_text[json_start + len('ELASTICSEARCH_QUERY:'):].strip()

                    # Remove any markdown code blocks
                    json_part = re.sub(r'^```json\s*', '', json_part, flags=re.MULTILINE)
                    json_part = re.sub(r'^```\s*', '', json_part, flags=re.MULTILINE)
                    json_part = re.sub(r'```\s*$', '', json_part, flags=re.MULTILINE)

                    # Find the JSON object
                    brace_count = 0
                    json_start_idx = -1
                    json_end_idx = -1

                    for i, char in enumerate(json_part):
                        if char == '{':
                            if json_start_idx == -1:
                                json_start_idx = i
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and json_start_idx != -1:
                                json_end_idx = i + 1
                                break

                    if json_start_idx != -1 and json_end_idx != -1:
                        json_string = json_part[json_start_idx:json_end_idx]
                        query_json = json.loads(json_string)

            except Exception as e:
                logger.error(f"Error parsing JSON: {e}")
                print(f"Failed to parse JSON from: {json_part}")

            # Post-process: Convert 'match' with 'type': 'phrase' to 'match_phrase'
            def fix_match_phrase(obj):
                if isinstance(obj, dict):
                    new_obj = {}
                    for k, v in obj.items():
                        # Check for 'match' with 'type': 'phrase'
                        if k == 'match' and isinstance(v, dict):
                            for field, match_val in v.items():
                                if isinstance(match_val, dict) and match_val.get('type') == 'phrase':
                                    # Convert to match_phrase
                                    new_obj['match_phrase'] = {field: match_val['query']}
                                else:
                                    new_obj[k] = v
                        else:
                            new_obj[k] = fix_match_phrase(v)
                    return new_obj
                elif isinstance(obj, list):
                    return [fix_match_phrase(i) for i in obj]
                else:
                    return obj

            query_json = fix_match_phrase(query_json)

            # Add a post-processing step to remove or correct any aggregation with "script" as the type
            def remove_invalid_script_aggs(obj):
                if isinstance(obj, dict):
                    new_obj = {}
                    for k, v in obj.items():
                        # If this is an aggregation with 'script' as the type, skip it
                        if k in ['aggs', 'aggregations'] and isinstance(v, dict):
                            new_obj[k] = {}
                            for agg_name, agg_body in v.items():
                                if isinstance(agg_body, dict) and 'script' in agg_body:
                                    # If 'script' is the only key, skip this aggregation
                                    if set(agg_body.keys()) == {'script'}:
                                        continue
                                    # If 'script' is used as a type, skip or fix
                                    if 'script' in agg_body and isinstance(agg_body['script'], dict):
                                        # Try to wrap it in a sum aggregation
                                        new_obj[k][agg_name] = {'sum': {'script': agg_body['script']}}
                                        continue
                                new_obj[k][agg_name] = remove_invalid_script_aggs(agg_body)
                        else:
                            new_obj[k] = remove_invalid_script_aggs(v)
                    return new_obj
                elif isinstance(obj, list):
                    return [remove_invalid_script_aggs(i) for i in obj]
                else:
                    return obj

            query_json = remove_invalid_script_aggs(query_json)

            # Fallback logic
            if not selected_index:
                if state['catalog_data']:
                    selected_index = list(state['catalog_data'].keys())[0]
                else:
                    selected_index = "total_exposures_index"  # Default fallback

            if not query_json:
                # Default query based on intent
                if state['intent'] == 'table':
                    query_json = {
                        "query": {"match_all": {}},
                        "size": 1000
                    }
                else:
                    query_json = {
                        "query": {"match_all": {}},
                        "size": 100
                    }

            # Ensure query has proper structure
            if 'query' not in query_json:
                query_json = {"query": query_json}

            # Add size limit if not present
            if 'size' not in query_json and 'aggs' not in query_json:
                query_json['size'] = 1000

            state['selected_index'] = selected_index
            state['elasticsearch_query'] = query_json

            logger.info(f"Generated query for index: {selected_index}")
            print(f"Final query: {json.dumps(query_json, indent=2)}")

        except Exception as e:
            logger.error(f"Error generating query: {e}")
            state['error'] = f"Error generating query: {str(e)}"

            # Fallback values
            if state['catalog_data']:
                state['selected_index'] = list(state['catalog_data'].keys())[0]
            else:
                state['selected_index'] = "total_exposures_index"

            state['elasticsearch_query'] = {
                "query": {"match_all": {}},
                "size": 1000
            }

        return state

    def fetch_data(self, state: AgentState) -> AgentState:
        """Fetch data from Elasticsearch"""
        try:
            logger.info(f"Fetching data from index: {state['selected_index']}")

            # Prepare the search parameters
            query_dict = state['elasticsearch_query']

            # Handle aggregations vs regular queries
            if 'aggs' in query_dict:
                # For aggregation queries
                search_params = {
                    'index': state['selected_index'],
                    'size': query_dict.get('size', 0),
                    'query': query_dict.get('query', {"match_all": {}}),
                    'aggregations': query_dict['aggs']
                }
            else:
                # For regular queries
                search_params = {
                    'index': state['selected_index'],
                    'query': query_dict.get('query', {"match_all": {}}),
                    'size': query_dict.get('size', 1000)
                }

                # Add _source if specified
                if '_source' in query_dict:
                    search_params['_source'] = query_dict['_source']

            print(f"Elasticsearch search parameters: {json.dumps(search_params, indent=2)}")

            # Execute the search
            response = self.es.search(**search_params)

            raw_data = []

            # Handle aggregation results
            if 'aggregations' in response:
                # Convert aggregation results to a more readable format
                aggs_data = []
                for agg_name, agg_result in response['aggregations'].items():
                    if 'buckets' in agg_result:
                        for bucket in agg_result['buckets']:
                            bucket_data = {'key': bucket['key'], 'doc_count': bucket['doc_count']}
                            # Add sub-aggregation results
                            for sub_agg_name, sub_agg_result in bucket.items():
                                if sub_agg_name not in ['key', 'doc_count']:
                                    if isinstance(sub_agg_result, dict) and 'value' in sub_agg_result:
                                        bucket_data[sub_agg_name] = sub_agg_result['value']
                            aggs_data.append(bucket_data)
                    else:
                        # Single metric aggregation
                        aggs_data.append({agg_name: agg_result.get('value', agg_result)})

                raw_data = aggs_data
            else:
                # Handle regular search results
                for hit in response['hits']['hits']:
                    raw_data.append(hit['_source'])

            state['raw_data'] = raw_data
            logger.info(f"Fetched {len(raw_data)} records")

            # Also store the raw ES response for debugging
            state['elasticsearch_response'] = response

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            state['error'] = f"Error fetching data: {str(e)}"
            state['raw_data'] = []

        return state

    def respond(self, state: AgentState) -> AgentState:
        """Generate final response based on intent"""
        try:
            if not state['raw_data']:
                state['final_response'] = "No data found for your query."
                return state

            if state['intent'] == 'table':
                # For table intent, just pass the data as is
                state['final_response'] = "Data retrieved successfully for table display."

            elif state['intent'] == 'chat':
                # Summarize data if too large
                summarized_data = self._summarize_data(state['raw_data'])
                state['summarized_data'] = summarized_data

                # Generate chat response using Ollama
                chat_response = self._generate_chat_response(
                    state['user_query'],
                    summarized_data
                )
                state['final_response'] = chat_response

            logger.info(f"Generated response for intent: {state['intent']}")

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state['error'] = f"Error generating response: {str(e)}"
            state['final_response'] = "Error generating response."

        return state

    def _format_catalog_data(self, catalog_data: Dict[str, List[str]]) -> str:
        """Format catalog data for prompt"""
        formatted = []
        for index_name, columns in catalog_data.items():
            columns_str = ", ".join(columns[:20])  # Limit columns to avoid token limits
            if len(columns) > 20:
                columns_str += f" ... (+{len(columns) - 20} more)"
            formatted.append(f"{index_name}: [{columns_str}]")
        return "\n".join(formatted)

    def _summarize_data(self, raw_data: List[Dict]) -> str:
        """Summarize data for chat context"""
        try:
            if not raw_data:
                return "No data available."

            df = pd.DataFrame(raw_data)

            summary = []
            summary.append(f"Dataset Summary: {len(df)} records, {len(df.columns)} columns")
            summary.append(f"Columns: {', '.join(df.columns.tolist())}")

            # Sample data (first 3 rows)
            summary.append("\nSample Records:")
            for i, row in df.head(3).iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                summary.append(f"Record {i + 1}: {row_str}")

            # Basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summary.append(f"\nNumeric Statistics:")
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    stats = df[col].describe()
                    summary.append(f"{col}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}")

            # Categorical summaries
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                if df[col].nunique() <= 10:
                    value_counts = df[col].value_counts().head(5)
                    summary.append(f"\n{col} distribution: {dict(value_counts)}")

            full_summary = "\n".join(summary)

            # Limit to 2000 words approximately
            if len(full_summary) > 8000:  # Rough estimation: 4 chars per word
                full_summary = full_summary[:8000] + "\n[Summary truncated to fit context limits]"

            return full_summary

        except Exception as e:
            logger.error(f"Error summarizing data: {e}")
            return f"Error processing data: {str(e)}"

    def _generate_chat_response(self, user_query: str, summarized_data: str) -> str:
        """Generate chat response using Ollama"""
        try:
            prompt = f"""
You are a helpful banking data analyst assistant. Based on the provided data summary, answer the user's question in a conversational and informative manner.

User's Question: {user_query}

Data Summary:
{summarized_data}

Instructions:
1. Provide a clear, concise answer to the user's question
2. Include relevant insights from the data
3. Use banking terminology appropriately
4. Keep the response conversational and helpful
5. Highlight key findings or patterns
6. If applicable, provide actionable recommendations

Response:"""

            response = self.ollama_llm.invoke(prompt)
            return response

        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"

    def process_query(self, user_query: str) -> Dict:
        """Process user query through the LangGraph workflow"""
        try:
            initial_state = AgentState(
                user_query=user_query,
                intent="",
                catalog_data={},
                elasticsearch_query={},
                selected_index="",
                raw_data=[],
                summarized_data="",
                final_response="",
                error="",
                elasticsearch_response={}
            )

            # Run the workflow
            final_state = self.app.invoke(initial_state)

            return {
                'success': True,
                'intent': final_state.get('intent'),
                'raw_data': final_state.get('raw_data'),
                'final_response': final_state.get('final_response'),
                'summarized_data': final_state.get('summarized_data'),
                'selected_index': final_state.get('selected_index'),
                'elasticsearch_query': final_state.get('elasticsearch_query'),
                'error': final_state.get('error')
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    st.set_page_config(
        page_title="Smart Banking Agent",
        page_icon="🏦",
        layout="wide"
    )

    st.title("🏦 Smart Banking Agent")
    st.markdown("*AI-powered banking data analysis with LangGraph, Groq, and Ollama*")
    st.markdown("---")

    # Hardcoded Groq API Key
    HARDCODED_GROQ_API_KEY = "api-key"

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'agent_ready' not in st.session_state:
        st.session_state.agent_ready = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'latest_result' not in st.session_state:
        st.session_state.latest_result = None

    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        # Catalog index input
        catalog_index = st.text_input(
            "Catalog Index Name",
            value="catalog_index",
            help="Name of the Elasticsearch index containing catalog data"
        )

        if st.session_state.agent_ready:
            st.success("✅ Agent: Ready")
        else:
            st.warning("⚠️ Agent: Not Ready")

        if st.button("🔄 Initialize Agent"):
            st.session_state.agent = None
            st.session_state.agent_ready = False
            st.rerun()

        st.markdown("---")
        st.markdown("**Intent Types:**")
        st.markdown("📊 **Table**: Display raw data")
        st.markdown("💬 **Chat**: AI analysis & insights")

        # Query history
        if st.session_state.query_history:
            st.subheader("📝 Recent Queries")
            for i, query in enumerate(st.session_state.query_history[-5:]):
                st.write(f"{i + 1}. {query}")

    # Initialize agent
    if not st.session_state.agent_ready:
        with st.spinner("Initializing Smart Banking Agent..."):
            try:
                # Test Groq API before initializing agent
                try:
                    test_llm = ChatGroq(
                        model="llama3-8b-8192",
                        temperature=0.1,
                        groq_api_key=HARDCODED_GROQ_API_KEY
                    )
                    test_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a test system."),
                        ("human", "ping")
                    ])
                    test_response = test_llm.invoke(test_prompt.format_messages())
                    if not test_response or not hasattr(test_response, 'content'):
                        raise Exception("No response from Groq API.")
                except Exception as api_e:
                    st.error(f"❌ Error connecting to Groq API: {api_e}")
                    st.stop()

                agent = SmartBankingAgent(HARDCODED_GROQ_API_KEY, catalog_index)
                st.session_state.agent = agent
                st.session_state.agent_ready = True
                st.success("🎉 Agent initialized successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error initializing agent: {e}")
                st.info("Please check your Groq API key, Elasticsearch, and Ollama connections.")
                st.stop()

    # Main interface
    if not st.session_state.agent_ready:
        st.info("🔄 Click 'Initialize Agent' in the sidebar to start")
        st.stop()

    # Query input
    with st.form("query_form"):
        user_query = st.text_area(
            "Ask me anything about your banking data:",
            height=100,
            placeholder="Examples:\n• Show me all customer transactions from last month\n• Analyze loan approval patterns and explain trends\n• What insights can you provide about account performance?"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.form_submit_button("🚀 Process Query", use_container_width=True, type="primary")
        with col2:
            clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True)

    if clear_button:
        st.session_state.latest_result = None
        st.rerun()

    # Process query
    if submit_button and user_query.strip():
        # Add to query history
        st.session_state.query_history.append(user_query.strip())

        with st.spinner("🔍 Processing your query through LangGraph workflow..."):
            try:
                result = st.session_state.agent.process_query(user_query)
                st.session_state.latest_result = result
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")

    # Display results
    result = st.session_state.latest_result
    if result:
        if not result.get('success', False):
            st.error(f"**Error:** {result.get('error', 'Unknown error')}")
        else:
            intent = result.get('intent', '')

            if intent == 'table':
                st.markdown("### 📊 Table Display Results")
                st.info(f"**Intent:** Table Display | **Records Found:** {len(result.get('raw_data', []))}")

                # Display data
                raw_data = result.get('raw_data', [])
                if raw_data:
                    df = pd.DataFrame(raw_data)
                    st.dataframe(df, use_container_width=True)

                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"banking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data found for your query.")

            elif intent == 'chat':
                st.markdown("### 💬 AI Analysis & Insights")
                st.info(f"**Intent:** Chat Analysis | **Records Analyzed:** {len(result.get('raw_data', []))}")

                # Display chat response
                final_response = result.get('final_response', '')
                if final_response:
                    st.markdown("#### 🤖 AI Response")
                    st.write(final_response)

                # Show data summary
                summarized_data = result.get('summarized_data', '')
                if summarized_data:
                    with st.expander("📊 Data Summary"):
                        st.text(summarized_data)

            # Query details
            with st.expander("🔧 Query Details"):
                st.json(result.get('elasticsearch_query', {}))
                st.write(f"**Selected Index:** {result.get('selected_index', 'N/A')}")

        st.markdown("---")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Smart Banking Agent** | "
        "Built with LangGraph, Groq, Ollama, Elasticsearch, and Streamlit | "
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )


if __name__ == "__main__":
    main()
