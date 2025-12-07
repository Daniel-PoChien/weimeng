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

# Page Setup
st.set_page_config(page_title="NBA AI Analyst", page_icon="🏀")
st.title("🏀 NBA Stats Agent")

# Sidebar for API Key and Mode Selection
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
        help="Auto: Tries Pandas Agent first, falls back to RAG on token/connection/parsing errors (if available)."
    )
    
    if mode == "Auto (Smart)":
        st.info("Will try Pandas Agent first, then RAG if needed and available.")
    elif mode == "Pandas Agent":
        st.info("Best for complex calculations over the CSV stats.")
    else:
        st.info("Best for semantic search over the CSV content (with general NBA knowledge).")

if api_key:
    # Load Data (If files exist)
    try:
        df_teams = pd.read_csv("nba_teams.csv")
        df_rosters = pd.read_csv("nba_rosters.csv")
        df_scores = pd.read_csv("nba_scores.csv")
        
        # Setup the Groq "Instant" AI Model
        llm = ChatGroq(
            temperature=0, 
            model_name="llama-3.1-8b-instant",
            groq_api_key=api_key
        )

        # Initialize both systems

        # Pandas Agent (for complex queries, calculations)
        pandas_agent = create_pandas_dataframe_agent(
            llm,
            [df_teams, df_rosters, df_scores],
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
            number_of_head_rows=3,  # Only show first 3 rows in prompt to reduce token usage
        )
        
        # RAG System (for semantic search + general NBA knowledge)
        try:
            retriever = get_retriever()

            # Hybrid RAG prompt: use context when helpful, otherwise use general NBA knowledge
            prompt_template = """
You are an NBA analyst. You have two sources of information:

1. Retrieved context from a local NBA stats database (teams, rosters, box scores, game results).
2. Your own general basketball knowledge about players, teams, awards, and history.

Use the context if it clearly contains relevant stats or facts.
If the context is missing, incomplete, or not directly related to the question,
you may also use your general NBA knowledge and reasonable judgment to answer.

If part of your answer is based mainly on general knowledge rather than the context,
briefly say so (for example: "Based on general NBA knowledge...").

If the question is extremely subjective or impossible to answer precisely,
explain the uncertainty but still give your best, well-reasoned answer.

Context:
{context}

Question:
{question}

Answer (2–4 sentences, clear and conversational):
"""
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
            st.sidebar.warning(
                "RAG not available (Chroma retriever failed):\n"
                f"{e}\n\n"
                "The app will still work using the Pandas Agent only.\n"
                "If this mentions chroma or pydantic, try rebuilding the DB or adjusting versions."
            )
            rag_available = False
            rag_chain = None

        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # New user input
        if prompt := st.chat_input("Ex: Which team has the most wins?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                answer = None
                used_mode = None
                error_occurred = False
                
                # Determine which systems to consider
                use_pandas = (mode == "Pandas Agent" or mode == "Auto (Smart)")
                # Only use RAG by default if user explicitly picked RAG.
                # Auto mode may switch to RAG later if needed and available.
                use_rag = (mode == "RAG")
                
                # --- 1) Try Pandas Agent first (Auto or Pandas Agent mode) ---
                if use_pandas and not use_rag:
                    with st.spinner("Using Pandas Agent..."):
                        try:
                            response = pandas_agent.invoke(prompt)
                            answer = response["output"]
                            used_mode = "Pandas Agent"
                        except Exception as e:
                            error_msg = str(e)
                            error_type = type(e).__name__
                            
                            # Check if it's a token/context/rate-limit error
                            if (
                                "413" in error_msg
                                or "token" in error_msg.lower()
                                or "context length" in error_msg.lower()
                                or "rate_limit" in error_msg.lower()
                            ):
                                if rag_available:
                                    error_occurred = True
                                    st.warning("Token/context limit reached! Auto-switching to RAG mode...")
                                    use_rag = True  # Fallback to RAG
                                else:
                                    st.warning(
                                        "Token/context limit reached, but RAG is not available.\n"
                                        "Try simplifying your question or narrowing the scope."
                                    )
                                    answer = None
                            # Check if it's a connection error
                            elif (
                                isinstance(e, APIConnectionError)
                                or "APIConnectionError" in error_type
                                or "Connection" in error_type
                                or "connection" in error_msg.lower()
                            ):
                                error_occurred = True
                                st.warning(
                                    "Groq API connection error. This could be due to:\n"
                                    "- Network connectivity issues\n"
                                    "- Groq API being temporarily unavailable\n"
                                    "- API key issues\n\n"
                                    "Auto-switching to RAG mode (if available)..."
                                )
                                if rag_available:
                                    use_rag = True  # Fallback to RAG
                                else:
                                    st.error(
                                        f"Connection Error: {error_msg}\n\n"
                                        "Please check your internet connection and Groq API status."
                                    )
                                    answer = None
                            else:
                                # Special case: LangChain OUTPUT_PARSING_FAILURE from the Pandas Agent
                                if (
                                    "OUTPUT_PARSING_FAILURE" in error_msg
                                    or "output parsing error occurred" in error_msg.lower()
                                ):
                                    error_occurred = True
                                    st.warning(
                                        "The Pandas stats agent got confused while parsing its own output.\n"
                                        "I'll try answering using the RAG (semantic search) mode instead."
                                    )
                                    if rag_available:
                                        use_rag = True  # we'll hit the RAG block next
                                    else:
                                        st.error(
                                            "Both the Pandas Agent and RAG are having trouble with this question.\n"
                                            "Try asking something more specific about stats in the dataset."
                                        )
                                        answer = None
                                else:
                                    # For other errors, show a short message
                                    st.error(
                                        "Error in Pandas Agent. "
                                        "Try rephrasing or narrowing your question."
                                    )
                                    # Uncomment below if you want the raw error in the UI:
                                    # st.text(error_msg)
                                    answer = None
                
                # --- 2) Use RAG (either explicitly selected or as fallback) ---
                if use_rag and rag_available:
                    with st.spinner("Using RAG (semantic search over CSV data + NBA knowledge)..."):
                        try:
                            # RetrievalQA default input key is "query"
                            response = rag_chain.invoke({"query": prompt})
                            answer = response["result"]
                            used_mode = (
                                "RAG"
                                if not error_occurred
                                else "RAG (Auto-fallback from Pandas Agent)"
                            )
                        except Exception as e:
                            st.error(f"RAG Error: {e}")
                            answer = None
                elif use_rag and not rag_available:
                    # User explicitly chose RAG, but it's not available.
                    # Fall back to Pandas Agent instead of just erroring.
                    with st.spinner("RAG not available. Using Pandas Agent instead..."):
                        try:
                            response = pandas_agent.invoke(prompt)
                            answer = response["output"]
                            used_mode = "Pandas Agent (RAG unavailable)"
                        except Exception as e:
                            st.error(f"RAG not available and Pandas Agent failed: {e}")
                            answer = None
                
                # --- 3) Display the answer ---
                if answer:
                    # Show which mode was used
                    if used_mode is None:
                        mode_badge = "🤖"
                        used_mode = "Unknown"
                    else:
                        mode_badge = (
                            "🤖" if "Auto" in used_mode
                            else "📊" if "Pandas" in used_mode
                            else "🔍"
                        )
                    st.caption(f"{mode_badge} Mode: {used_mode}")
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                elif not error_occurred:
                    st.error("Failed to get an answer. Please try again.")

    except FileNotFoundError:
        st.error("CSV files not found! Please run 'python get_data.py' first.")
else:
    st.info("Please enter your API Key to start.")

