import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from rag_engine import get_retriever
import os

# Import Groq errors for better error handling
try:
    from groq import APIConnectionError, APIError
except ImportError:
    APIConnectionError = Exception
    APIError = Exception

# 1. Page Setup
st.set_page_config(page_title="NBA AI Analyst", page_icon="🏀")
st.title("🏀 NBA Stats Agent")

# 2. Sidebar for API Key and Mode Selection
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Groq API Key:", type="password")
    st.markdown("[Get a Free Key Here](https://console.groq.com/keys)")
    
    st.divider()
    st.header("Query Mode")
    # Mode selector: Auto (smart fallback), Pandas Agent, or RAG
    mode = st.radio(
        "Choose query mode:",
        ["Auto (Smart)", "Pandas Agent", "RAG"],
        help="Auto: Tries Pandas Agent first, falls back to RAG on errors"
    )
    
    if mode == "Auto (Smart)":
        st.info("Will auto-switch to RAG if token limit reached")
    elif mode == "Pandas Agent":
        st.info("Best for complex calculations")
    else:
        st.info("Best for semantic search (lower token usage)")

if api_key:
    # 3. Load Data (If files exist)
    try:
        df_teams = pd.read_csv("nba_teams.csv")
        df_rosters = pd.read_csv("nba_rosters.csv")
        df_scores = pd.read_csv("nba_scores.csv")
        
        # 4. Setup the "Instant" AI Model (Fixes Rate Limit Error)
        llm = ChatGroq(
            temperature=0, 
            model_name="llama-3.1-8b-instant",
            groq_api_key=api_key
        )

        # 5. Initialize both systems
        # Pandas Agent (for complex queries)
        pandas_agent = create_pandas_dataframe_agent(
            llm, 
            [df_teams, df_rosters, df_scores], 
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            number_of_head_rows=3  # Only show first 3 rows in prompt to reduce token usage
        )
        
        # RAG System (for semantic search, lower token usage)
        try:
            retriever = get_retriever()
            # Create a prompt template for RAG
            prompt_template = """Use the following pieces of context to answer the question about NBA statistics.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}

            Question: {question}

            Answer:"""
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=False
            )
            rag_available = True
        except Exception as e:
            st.sidebar.warning(f"RAG not available: {e}\n\nTo fix: pip install chromadb\nThen build: python -c \"from rag_engine import build_database; build_database()\"")
            rag_available = False
            rag_chain = None

        # 6. Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ex: Which team has the most wins?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                answer = None
                used_mode = None
                error_occurred = False
                
                # Determine which mode to use
                use_pandas = (mode == "Pandas Agent" or mode == "Auto (Smart)")
                use_rag = (mode == "RAG" or (mode == "Auto (Smart)" and not rag_available))
                
                # Try Pandas Agent first (if selected)
                if use_pandas and not use_rag:
                    with st.spinner("Using Pandas Agent..."):
                        try:
                            response = pandas_agent.invoke(prompt)
                            answer = response['output']
                            used_mode = "Pandas Agent"
                        except Exception as e:
                            error_msg = str(e)
                            error_type = type(e).__name__
                            
                            # Check if it's a token limit error
                            if "413" in error_msg or "token" in error_msg.lower() or "rate_limit" in error_msg.lower():
                                error_occurred = True
                                st.warning("Token limit reached! Auto-switching to RAG mode...")
                                use_rag = True  # Fallback to RAG
                            # Check if it's a connection error
                            elif isinstance(e, APIConnectionError) or "APIConnectionError" in error_type or "Connection" in error_type or "connection" in error_msg.lower():
                                error_occurred = True
                                st.warning("Groq API connection error. This could be due to:\n- Network connectivity issues\n- Groq API being temporarily unavailable\n- API key issues\n\nAuto-switching to RAG mode...")
                                if rag_available:
                                    use_rag = True  # Fallback to RAG
                                else:
                                    st.error(f"Connection Error: {error_msg}\n\nPlease check your internet connection and Groq API status.")
                                    answer = None
                            else:
                                # For other errors, show the error but don't crash
                                st.error(f"Error: {error_msg}")
                                answer = None
                
                # Use RAG (either selected or as fallback)
                if use_rag and rag_available:
                    with st.spinner("Using RAG (semantic search)..."):
                        try:
                            response = rag_chain.invoke({"query": prompt})
                            answer = response['result']
                            used_mode = "RAG" if not error_occurred else "RAG (Auto-fallback)"
                        except Exception as e:
                            st.error(f"RAG Error: {e}")
                            answer = None
                elif use_rag and not rag_available:
                    st.error("RAG system not available. Please build the database first.")
                    answer = None
                
                # Display the answer
                if answer:
                    # Show which mode was used
                    mode_badge = "🤖" if "Auto" in used_mode else "📊" if "Pandas" in used_mode else "🔍"
                    st.caption(f"{mode_badge} Mode: {used_mode}")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                elif not error_occurred:
                    st.error("Failed to get an answer. Please try again.")

    except FileNotFoundError:
        st.error(" CSV files not found! Please run 'python get_data.py' first.")
else:
    st.info("Please enter your API Key to start.")