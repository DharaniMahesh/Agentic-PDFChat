import os
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import hashlib
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
import autogen
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain.memory import ConversationBufferMemory
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import google.generativeai as genai 
# import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.embeddings.google_palm import GooglePalmEmbeddings
from langchain_community.embeddings.google_palm import GooglePalmEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import SentenceTransformer
import torch
from autogen.agentchat.contrib.web_surfer import WebSurferAgent
from duckduckgo_search import DDGS
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq as Gr

load_dotenv()
access_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
# print(access_token  + "-----------------------------------------------------------------------------------------")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class DocumentProcessor:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_pdf_text(self, pdf_docs) -> str:
        text = ""
        total_pages = 0
        for pdf in pdf_docs:
            reader = PdfReader(pdf)
            total_pages += len(reader.pages)
            for page in reader.pages:
                text += page.extract_text()
        st.sidebar.info(f"Processed {len(pdf_docs)} documents, {total_pages} pages total")
        return text

    def get_document_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get_cached_embeddings(self, doc_hash: str, embeddings) -> Optional[FAISS]:
        cache_file = self.cache_dir / f"{doc_hash}.faiss"
        if cache_file.exists():
            try:
                return FAISS.load_local(
                    folder_path=str(cache_file),
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True  # Added this parameter
                )
            except Exception as e:
                st.warning(f"Cache loading failed: {str(e)}. Regenerating embeddings...")
                return None
        return None

    def cache_embeddings(self, doc_hash: str, vectorstore: FAISS):
        cache_file = self.cache_dir / f"{doc_hash}.faiss"
        vectorstore.save_local(str(cache_file))

    def clear_cache(self):
        """Clear all cached embeddings"""
        try:
            for file in self.cache_dir.glob("*.faiss"):
                file.unlink()
            st.success("Cache cleared successfully")
        except Exception as e:
            st.error(f"Error clearing cache: {str(e)}")



    def cache_embeddings(self, doc_hash: str, vectorstore: FAISS):
        """Save embeddings to cache"""
        cache_file = self.cache_dir / f"{doc_hash}.faiss"
        vectorstore.save_local(str(cache_file))


class QASystem:
    def __init__(self):
        self.groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.conversation_history = []

    def get_groq_response(self, prompt: str) -> str:
        """Get response from Groq API"""
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specializing in document analysis and providing detailed, accurate information."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="Llama3-70b-8192",
                temperature=0.3,
                max_tokens=512
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error in Groq API call: {str(e)}")
            return "Error processing request"


class GroqWrapper:
    """Wrapper class for Groq client that implements deepcopy"""

    def __init__(self, api_key):
        self.api_key = api_key
        self.client = Groq(api_key=api_key)

    def __deepcopy__(self, memo):
        """Implement deepcopy method"""
        return GroqWrapper(self.api_key)

    def create(self, messages, model="Llama3-70b-8192", temperature=0.3, max_tokens=512):
        """Create a chat completion"""
        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")


class AutoGenAgents:
    def __init__(self, config_list):
        self.config_list = config_list
        self.groq_wrapper = GroqWrapper(api_key=os.getenv("GROQ_API_KEY"))
        self.setup_agents()

    def setup_agents(self):
        # Base LLM configuration
        load_dotenv()
        x_config_list = [{
            "model": "llama-3.3-70b-versatile",
            "api_key": os.environ.get("GROQ_API_KEY"),
            "api_type": "groq"
        }]

        # Web surfer configuration for fact checker
        web_surfer_config = {
            "timeout": 600,
            "cache_seed": None,
            "config_list": self.config_list,
            "temperature": 0,
        }

        # Browser configuration for web search
        browser_config = {
            "viewport_size": 4096,
            "bing_api_key": os.getenv("BING_API_KEY", "default_key")
        }

        code_execution_config = {
            "work_dir": "coding",
            "use_docker": False  # Explicitly disable Docker
        }
        # Primary assistant for general queries
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            system_message="Coordinate between different agents to get comprehensive answers.",
            human_input_mode="NEVER",
            code_execution_config=code_execution_config
        )

        # Primary assistant for general queries
        self.assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config={"config_list": x_config_list},
            system_message="""You are a helpful assistant specializing in analyzing documents 
            and providing initial analysis of the content."""
        )

        # Research agent for deep analysis
        self.researcher = autogen.AssistantAgent(
            name="researcher",
            llm_config={"config_list": x_config_list},
            system_message="""You are a research specialist who excels at in-depth analysis 
            of documents. Focus on identifying patterns and extracting key insights."""
        )

        # Web surfer agent for fact checking and web references
        self.web_surfer = WebSurferAgent(
            name="web_surfer",
            llm_config={"config_list": x_config_list},
            summarizer_llm_config={"config_list": x_config_list},
            browser_config={
                "viewport_size": 4096,
                "bing_api_key": os.getenv("BING_API_KEY", "1111")
            },
        )

    async def process_query(self, query: str, context: str, analysis_type: str) -> Dict:
        """Process a query using multiple agents and return the result."""

        async def get_groq_response(messages):
            """Helper function to get refined response from Groq."""
            try:
                response = self.groq_wrapper.create(
                    messages=messages,
                    model="Llama3-70b-8192",
                    temperature=0.3,
                    max_tokens=512
                )
                return response.choices[0].message.content
            except Exception as e:
                st.error(f"Error in Groq API call: {str(e)}")
                return "Error processing request."

        # Initialize chat history for tracking conversation
        chat_history = []

        # Prepare the system context message
        context_message = {
            "role": "system",
            "content": f"""TASK: {analysis_type.upper()} QUERY
            CONTEXT: {context}

            <<<Please analyze this content which is extracted from the user uploaded document thoroughly and provide a comprehensive response mostly based on the GIVEN CONTEXT so that it would be like your answer is generated from the uploaded document.>>>"""
        }
        chat_history.append(context_message)

        try:
            # Step 1: Start conversation with assistant
            # st.write("Starting conversation with Assistant Agent...")
            assistant_response = self.user_proxy.initiate_chat(
                self.assistant,
                message=f'''Question: {query}\n<<<Please provide initial analysis in 10 lines>>>
                            Current Context: {context}
                            Analysis Type: {analysis_type}
                        ''',
                max_turns=1
            )
            # print(assistant_response)
            # if not assistant_response or "content" not in assistant_response:
            #     raise ValueError("Assistant failed to provide a valid response.")
            try:
                assistant_content = assistant_response.chat_history[-1]["content"]
            except:
                print("error near assitant_content")

            chat_history.append({"role": "assistant", "content": assistant_content})
            # st.write("Assistant response received.")

            # Step 2: Continue with researcher for detailed analysis
            # st.write("Initiating conversation with Researcher Agent...")
            researcher_response = self.user_proxy.initiate_chat(
                self.researcher,
                message=f'''Question: {query}\n<<<Please provide detailed research analysis in 20 lines and TERMINATE>>>
                            Actual Context: {context}
                            Analysis Type: {analysis_type}
                        ''',
                max_turns=1
            )
            
            # if not researcher_response or "content" not in researcher_response:
            #     raise ValueError("Researcher failed to provide a valid response.")
            
            researcher_content = researcher_response.chat_history[-1]["content"]
            chat_history.append({"role": "researcher", "content": researcher_content})
            # st.write("Researcher response received.")
            # print(assistant_content)
            # print(researcher_content)
            # Step 3: Combine and refine insights
            combined_analysis = f"""
            Initial Analysis:
            {assistant_content}

            Research Analysis:
            {researcher_content}

            Use this above analysis as reference, but make sure that your answer is based mostly on the given CONTEXT.
            """

            # Get refined analysis from Groq
            # st.write("Refining response using Groq API...")
            refined_response = await get_groq_response([context_message, {"role": "user", "content": combined_analysis}])
            print(refined_response)
            if not refined_response:
                raise ValueError("Refinement step failed to return a valid response.")

            chat_history.append({"role": "assistant", "content": refined_response})


        # Step 4: Perform web search for references
            web_references = []
            try:
                st.write("Initiating web search using phi's DuckDuckGo tool...")

                # Create an agent with DuckDuckGo as a tool
                agent_role = """
                    You are a web search agent. Your primary responsibility is to search the web for accurate and reliable information and return links I'l  also
                    provide a little infromation about the uploaded document for better search results
                    """ + '\n' + assistant_content

                    # Create the Web Search Agent
                web_search_agent = Agent(
                    name="Web Search Agent",
                    role=agent_role,  # Role description for the agent
                    model=Gr(  # Groq-based LLM for processing      # Model ID
                        id='Llama3-70b-8192',
                        temperature=0.1,       
                        max_tokens=2048,           
                        top_p=0.95,
                        api_key=os.getenv("GROQ_API_KEY") 
                    ),
                    tools=[
                        DuckDuckGo(  # Integrate the DuckDuckGo search tool
                            fixed_max_results=3,  
                            timeout=60            # Set a timeout of 60 seconds
                        ),
                    ],
                    show_tool_calls=True,  # Enable tool call logs for debuggin
                    instructions="also provide sources"
                )
                response = web_search_agent.run(f"Search the web for: {assistant_content}")

                messages = response.messages
                for message in messages:
                    # Use dot notation to access attributes of the Message object
                    if message.role == "tool" and "duckduckgo_search" in (message.tool_name or ""):
                        tool_content = message.content or "[]"
                        results = eval(tool_content)  # Convert stringified list to Python list

                        # Extract titles and links
                        web_references = [(item["title"], item["href"]) for item in results]

            except Exception as e:
                st.warning(f"Web search encountered an error: {str(e)}")


            # Step 5: Aggregate and return final response
            st.write("Aggregating final response...")
            final_response = {
                "response": refined_response,
                "web_references": web_references,
                "chat_history": chat_history,
                "agents_involved": ["user_proxy", "assistant", "researcher", "web_surfer"]
            }

            return final_response

        except Exception as e:
            st.error(f"Error during query processing: {str(e)}")
            return {
                "response": "Error processing request.",
                "chat_history": chat_history,
                "agents_involved": [],
                "web_references": []
            }


class EnhancedQASystem(QASystem):
    def __init__(self, config_list):
        super().__init__()
        self.autogen_agents = AutoGenAgents(config_list)
        self.total_tokens_used = 0
        self.query_count = 0
        self._conversation_history = []

    async def process_query(self, query: str, docs: List, analysis_type: str) -> Dict:
        context = "\n".join([doc.page_content for doc in docs])

        try:
            result = await self.autogen_agents.process_query(query, context, analysis_type)
            estimated_tokens = len(result['response'].split()) * 1.3

            self.total_tokens_used += estimated_tokens
            self.query_count += 1

            # Add web references to the display
            display_response = f"""
            {result['response']}
            
            Relevant Web References:
            """
            if result['web_references']:
                for idx, url in enumerate(result['web_references'], 1):
                    display_response += f"\n{idx}. {url}"
            else:
                display_response += "\nNo web references found."

            result.update({
                "response": display_response,
                "tokens_used": estimated_tokens,
                "total_tokens": self.total_tokens_used,
                "query_count": self.query_count
            })

            return result

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return {
                "response": "Error processing request",
                "chat_history": [],
                "agents_involved": [],
                "tokens_used": 0,
                "total_tokens": self.total_tokens_used,
                "query_count": self.query_count,
                "web_references": []
            }

class Analytics:
    @staticmethod
    def get_usage_stats(conversation_history: List[Dict]) -> pd.DataFrame:
        if not conversation_history:
            return pd.DataFrame()

        df = pd.DataFrame(conversation_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    @staticmethod
    def create_usage_charts(df: pd.DataFrame):
        if df.empty:
            return None, None, None, None

        # Color scheme
        bar_color = '#1f77b4'  # Blue
        line_color = '#2ca02c'  # Green
        grid_color = 'rgba(211, 211, 211, 0.5)'  # Light gray with transparency
        
        # Questions per day
        daily_questions = (df.groupby(df['timestamp'].dt.date)
                        .size()
                        .reset_index(name='count'))
        fig1 = px.bar(daily_questions,
                    x='timestamp',
                    y='count',
                    title='Questions Per Day',
                    labels={'count': 'Number of Questions', 'timestamp': 'Date'})
        fig1.update_traces(
            marker_color=bar_color,
            hovertemplate="<b>Date</b>: %{x}<br>" +
                        "<b>Questions</b>: %{y}<extra></extra>"
        )

        # Token usage over time
        fig2 = px.line(df,
                    x='timestamp',
                    y='tokens_used',
                    title='Token Usage Over Time')
        fig2.update_traces(
            line_color=line_color,
            hovertemplate="<b>Time</b>: %{x}<br>" +
                        "<b>Tokens</b>: %{y}<extra></extra>"
        )

        # Analysis type distribution
        analysis_counts = df['analysis_type'].value_counts().reset_index()
        analysis_counts.columns = ['analysis_type', 'count']
        fig3 = px.pie(analysis_counts,
                    values='count',
                    names='analysis_type',
                    title='Analysis Type Distribution')
        fig3.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>Type</b>: %{label}<br>" +
                        "<b>Count</b>: %{value}<br>" +
                        "<b>Percentage</b>: %{percent}<extra></extra>"
        )

        # Agent involvement
        agent_list = []
        for agents in df['agents_involved']:
            agent_list.extend(agents)
        agent_counts = (pd.Series(agent_list)
                    .value_counts()
                    .reset_index())
        agent_counts.columns = ['agent', 'count']
        fig4 = px.bar(agent_counts,
                    x='agent',
                    y='count',
                    title='Agent Participation',
                    labels={'count': 'Number of Queries', 'agent': 'Agent'})
        fig4.update_traces(
            marker_color=bar_color,
            hovertemplate="<b>Agent</b>: %{x}<br>" +
                        "<b>Queries</b>: %{y}<extra></extra>"
        )

        # Enhanced layout settings for all charts
        layout_updates = dict(
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='white',
            margin=dict(t=50, l=50, r=50, b=50),  # Increased margins for axis labels
            showlegend=True,
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="black"
            ),
            title=dict(
                font=dict(size=16, color='black'),
                x=0.5,
                xanchor='center'
            )
        )

        for fig in [fig1, fig2, fig3, fig4]:
            fig.update_layout(**layout_updates)
            
            # Update axes for better readability (except for pie chart)
            if fig != fig3:
                fig.update_xaxes(
                    showgrid=True,
                    gridcolor=grid_color,
                    gridwidth=1,
                    tickangle=45,
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    tickfont=dict(
                        size=14,
                        color='black'
                    ),
                    title_font=dict(
                        size=14,
                        color='black'
                    ),
                    title_standoff=25  # Add space between axis and title
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridcolor=grid_color,
                    gridwidth=1,
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    tickfont=dict(
                        size=14,
                        color='black'
                    ),
                    title_font=dict(
                        size=14,
                        color='black'
                    ),
                    title_standoff=25  # Add space between axis and title
                )
                
                # Add value labels on bars/points
                if isinstance(fig.data[0], go.Bar):
                    fig.update_traces(
                        textposition='outside',
                        texttemplate='%{y}',
                        textfont=dict(
                            size=12,
                            color='black'
                        )
                    )
                elif isinstance(fig.data[0], go.Scatter):
                    fig.update_traces(
                        mode='lines+markers+text',
                        textposition='top center',
                        texttemplate='%{y}',
                        textfont=dict(
                            size=12,
                            color='black'
                        )
                    )

                # Ensure ticks and tick labels are visible
                fig.update_xaxes(
                    ticks="outside",
                    tickwidth=2,
                    tickcolor='black',
                    ticklen=10
                )
                fig.update_yaxes(
                    ticks="outside",
                    tickwidth=2,
                    tickcolor='black',
                    ticklen=10
                )

        return fig1, fig2, fig3, fig4

    @staticmethod
    def display_analytics(df: pd.DataFrame, container):
        """Display analytics in the given Streamlit container"""
        if df.empty:
            container.info("No data available for analytics yet.")
            return

        fig1, fig2, fig3, fig4 = Analytics.create_usage_charts(df)
        
        if all([fig1, fig2, fig3, fig4]):  # Check if all figures were created
            container.plotly_chart(fig1, use_container_width=True)
            container.plotly_chart(fig2, use_container_width=True)
            
            # Create two columns for the pie chart and bar chart
            col1, col2 = container.columns(2)
            with col1:
                container.plotly_chart(fig3, use_container_width=True)
            with col2:
                container.plotly_chart(fig4, use_container_width=True)
            
            # Display summary statistics
            container.subheader("Summary Statistics")
            stats = {
                "Total Questions": len(df),
                "Total Tokens Used": df['tokens_used'].sum(),
                "Average Tokens per Query": df['tokens_used'].mean(),
                "Most Common Analysis Type": df['analysis_type'].mode().iloc[0],
                "Number of Unique Agents": len(set([
                    agent for agents in df['agents_involved'] for agent in agents
                ]))
            }
            
            # Display stats in a nice format
            stat_cols = container.columns(3)
            for i, (key, value) in enumerate(stats.items()):
                stat_cols[i % 3].metric(
                    key,
                    f"{value:,.0f}" if isinstance(value, (int, float)) else value
                )


def display_conversation_history(conv_history: List[Dict]):
    """Display conversation history with an enhanced UI"""
    st.subheader("Conversation History")

    for conv in reversed(conv_history[-5:]):
        with st.expander(f"Q: {conv['question'][:50]}...", expanded=False):
            st.markdown("**Question:**")
            st.write(conv['question'])

            st.markdown("**Response:**")
            st.write(conv['response'])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Analysis Type:**")
                st.info(conv['analysis_type'])

            with col2:
                st.markdown("**Agents Involved:**")
                st.success(", ".join(conv['agents_involved']))

            with col3:
                st.markdown("**Tokens Used:**")
                st.warning(f"{conv['tokens_used']:,}")

            st.markdown("**Timestamp:**")
            st.text(conv['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))


async def main():
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        st.error("Please set your GROQ_API_KEY in the .env file")
        return

    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Configure API keys
    config_list = [
        {
            'model': 'Llama3-70b-8192',
            'api_key': os.getenv("GROQ_API_KEY")
        }
    ]

    #st.set_page_config(page_title="Advanced PDF QA System with AutoGen", layout="wide")

    # Initialize components
    doc_processor = DocumentProcessor()
    qa_system = EnhancedQASystem(config_list)

     # Initialize qa_system conversation history from session state
    qa_system.conversation_history = st.session_state.conversation_history

    # Sidebar for uploading and settings
    with st.sidebar:
        st.header("Settings & Upload")
        pdfs = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

        analysis_type = st.selectbox(
            "Analysis Type",
            ["standard", "summary", "analysis", "deep_research", "fact_check"],
            help="Select the type of analysis you want to perform on the documents"
        )

        # if st.button("Clear Conversation History"):
        #     st.session_state.conversation_history = []
        #     qa_system.conversation_history = []
        #     st.experimental_rerun()

        st.subheader("Agent Settings")
        show_agent_conversation = st.checkbox(
            "Show Agent Conversations",
            value=False,
            help="Show the detailed conversation between different AI agents"
        )

        if st.button("Clear Conversation History"):
            qa_system.conversation_history = []
            st.session_state.clear()
            st.experimental_rerun()

    # Main content area
    st.title("Advanced PDF Question-Answering System with AutoGen")

    if pdfs:
        # Process documents
        with st.spinner("Processing documents..."):
            text = doc_processor.get_pdf_text(pdfs)
            doc_hash = doc_processor.get_document_hash(text)
            vectorstore = doc_processor.get_cached_embeddings(doc_hash, embeddings)

            if vectorstore is None:
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=500,  # Reduced from 1000
                    chunk_overlap=100,  # Reduced from 200
                    length_function=len
                )
                chunks = text_splitter.split_text(text)

                with st.spinner("Creating embeddings... This might take a while."):
                    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
                    vectorstore = FAISS.from_texts(
                        texts=chunks,
                        embedding=embeddings  # Pass the embeddings instance directly
                    )
                    doc_processor.cache_embeddings(doc_hash, vectorstore)

        # Create main content columns
        #col1, col2 = st.columns([2, 1])

        
        st.header("Ask Questions")
        user_query = st.text_input("Enter your question:")

        if user_query:
            docs = vectorstore.similarity_search(user_query)

            with st.spinner("Processing your question..."):
                start_time = datetime.now()
                result = await qa_system.process_query(user_query, docs, analysis_type)
                processing_time = (datetime.now() - start_time).total_seconds()

                # Create new conversation entry
                new_conversation = {
                    'question': user_query,
                    'response': result['response'],
                    'analysis_type': analysis_type,
                    'agents_involved': result['agents_involved'],
                    'tokens_used': result['tokens_used'],
                    'timestamp': datetime.now()
                }

                # Update both qa_system and session state conversation history
                st.session_state.conversation_history.append(new_conversation)
                qa_system.conversation_history = st.session_state.conversation_history

                # Display response
                st.markdown("### Answer")
                st.write(result['response'])

                # Display metrics
                metrics_cols = st.columns(4)
                metrics_cols[0].metric("Processing Time", f"{processing_time:.2f}s")
                metrics_cols[1].metric("Tokens Used", f"{result['tokens_used']:,}")
                metrics_cols[2].metric("Total Queries", len(st.session_state.conversation_history))
                metrics_cols[3].metric("Agents Used", len(result['agents_involved']))

                # Show agent conversations if enabled
                if show_agent_conversation:
                    st.markdown("### Agent Conversations")
                    for msg in result['chat_history']:
                        with st.chat_message(msg['role']):
                            st.write(msg['content'])

                # Update conversation history
                # qa_system.conversation_history.append({
                #     'question': user_query,
                #     'response': result['response'],
                #     'analysis_type': analysis_type,
                #     'agents_involved': result['agents_involved'],
                #     'tokens_used': result['tokens_used'],
                #     'timestamp': datetime.now()
                # })

        
        st.header("Analytics")
        if st.session_state.conversation_history:
            df = Analytics.get_usage_stats(st.session_state.conversation_history)
            Analytics.display_analytics(df, st)
        else:
            st.info("Start asking questions to see analytics!")

        # Display conversation history
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("---")
            st.header("Conversation History")
            
            for i, conv in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Q{len(st.session_state.conversation_history)-i}: {conv['question'][:50]}...", expanded=False):
                    st.markdown("**Question:**")
                    st.write(conv['question'])
                    
                    st.markdown("**Response:**")
                    st.write(conv['response'])
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown("**Analysis Type:**")
                        st.info(conv['analysis_type'])
                    
                    with cols[1]:
                        st.markdown("**Agents:**")
                        st.success(", ".join(conv['agents_involved']))
                    
                    with cols[2]:
                        st.markdown("**Tokens:**")
                        st.warning(f"{conv['tokens_used']:,}")
                    
                    with cols[3]:
                        st.markdown("**Time:**")
                        st.text(conv['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
# token_usage = response.response_metadata["token_usage"]
# input_tokens = token_usage["prompt_tokens"]
# output_tokens = token_usage["completion_tokens"]

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())