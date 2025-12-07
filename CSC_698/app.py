import streamlit as st
import pandas as pd
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from rag_engine import get_retriever
from live_stats import (
    get_player_season_average,
    format_season_average_human,
    LiveStatsError,
)

# Import Groq errors (used in some error handling logic)
try:
    from groq import APIConnectionError, APIError
except ImportError:
    APIConnectionError = Exception
    APIError = Exception


# --------------------------------------------------------
# 1. STREAMLIT PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(page_title="NBA AI Analyst", page_icon="🏀")
st.title("🏀 NBA Stats Agent")


# --------------------------------------------------------
# 2. SIDEBAR
# --------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Groq API Key:", type="password")
    st.markdown("[Get a Free Key Here](https://console.groq.com/keys)")

    st.divider()
    st.header("Query Mode")
    mode = st.radio(
        "Choose query mode:",
        ["Auto (Smart)", "Pandas Agent", "RAG"],
        help="Auto: Live API + RAG for general questions, Pandas for stats-heavy questions.",
    )

    if mode == "Auto (Smart)":
        st.info("Auto: Live API for 'this season' stats, RAG for concepts, Pandas for CSV stats.")
    elif mode == "Pandas Agent":
        st.info("Best for detailed calculations on the CSV stats.")
    else:
        st.info("Best for semantic search + general NBA knowledge.")


# --------------------------------------------------------
# 3. CACHE DATA FOR SPEED
# --------------------------------------------------------
@st.cache_data
def load_data():
    df_teams = pd.read_csv("nba_teams.csv")
    df_rosters = pd.read_csv("nba_rosters.csv")
    df_scores = pd.read_csv("nba_scores.csv")
    return df_teams, df_rosters, df_scores


# --------------------------------------------------------
# 4. QUESTION CLASSIFIERS (for Auto routing)
# --------------------------------------------------------
def is_beginner_question(q: str) -> bool:
    """Heuristic: sounds like a general / conceptual / beginner question?"""
    q = q.lower()

    beginner_phrases = [
        "who is", "what is", "explain", "tell me", "why",
        "best player", "good", "bad", "compare", "how good",
        "how tall", "who won", "history", "summary",
        "i don't know anything about basketball",
        "i dont know anything about basketball",
        "i don't know anything about the nba",
        "i dont know anything about the nba",
        "i know nothing about basketball",
        "i know nothing about the nba",
        "i'm new to basketball",
        "im new to basketball",
        "i'm new to the nba",
        "im new to the nba",
        "explain basketball",
        "explain the nba",
    ]
    if any(b in q for b in beginner_phrases):
        return True

    # If no stats keywords appear → assume conceptual question
    stats_keywords = [
        "pts", "points", "reb", "rebounds", "ast", "assists",
        "ppg", "apg", "rpg", "average", "mean", "median",
        "sum", "total", "max", "min", "highest", "lowest",
        "per game", "field goal", "3pt", "3-point", "efficiency", "stat", "stats",
    ]
    if not any(k in q for k in stats_keywords):
        return True

    return False


def looks_like_live_stats_question(q: str) -> bool:
    """
    Heuristic: questions about what someone is averaging *this season* / now.
    These should try the live API first.
    """
    q = q.lower()
    triggers = [
        "this season",
        "current season",
        "right now",
        "currently averaging",
        "average this year",
        "averaging this year",
        "stats this season",
        "averaging this season",
        "this year stats",
        "this year average",
    ]
    return any(t in q for t in triggers)


def extract_player_name_simple(q: str) -> str:
    """
    Very simple heuristic to detect player name in the question.
    For demo: if it mentions 'curry', 'lebron', etc.
    You can extend this later.
    """
    q_lower = q.lower()
    if "curry" in q_lower:
        return "Stephen Curry"
    if "lebron" in q_lower or "lebron james" in q_lower:
        return "LeBron James"
    if "giannis" in q_lower:
        return "Giannis Antetokounmpo"
    if "jokic" in q_lower or "jokić" in q_lower:
        return "Nikola Jokic"
    # Fallback: return empty string (unknown)
    return ""


# --------------------------------------------------------
# 5. MAIN APP LOGIC
# --------------------------------------------------------
if api_key:
    try:
        # Load data once (cached by Streamlit)
        df_teams, df_rosters, df_scores = load_data()

        # --------------------------------------------------------
        # 6. LLM SETUP
        # --------------------------------------------------------
        llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            groq_api_key=api_key,
        )

        # --------------------------------------------------------
        # 7. PANDAS AGENT (for stats / calculations)
        # --------------------------------------------------------
        pandas_agent = create_pandas_dataframe_agent(
            llm,
            [df_teams, df_rosters, df_scores],
            verbose=False,                  # less logging for speed
            allow_dangerous_code=True,
            agent_executor_kwargs={
                "handle_parsing_errors": True,  # let it recover from weird outputs
            },
            number_of_head_rows=3,
        )

        # --------------------------------------------------------
        # 8. RAG SYSTEM (Chroma + NBA CSVs + general NBA knowledge)
        # --------------------------------------------------------
        try:
            retriever = get_retriever()

            rag_prompt = """
You are an expert NBA analyst and teacher. The user might be a beginner.

You have two sources of information:
1. Retrieved context from a local NBA stats database (teams, rosters, box scores, game results).
2. Your own general basketball knowledge about players, teams, awards, and history.

Use the context if it clearly contains relevant stats or facts.
If the context is missing or incomplete, you may also use your general NBA knowledge.

If part of your answer is mainly based on general knowledge, briefly say so
(for example: "Based on general NBA knowledge...").

Explain things clearly. If the question sounds like it's from a beginner,
avoid heavy jargon or explain any basketball terms you use.

Context:
{context}

Question:
{question}

Answer (2–5 sentences, clear and conversational):
"""
            PROMPT = PromptTemplate(
                template=rag_prompt,
                input_variables=["context", "question"],
            )

            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=False,
            )
            rag_available = True
        except Exception as e:
            st.sidebar.error(
                "RAG not available:\n"
                f"{e}\n\n"
                "Make sure you built the database with:\n"
                'python -c "from rag_engine import build_database; build_database()"'
            )
            rag_available = False
            rag_chain = None

        # --------------------------------------------------------
        # 9. CHAT HISTORY
        # --------------------------------------------------------
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # --------------------------------------------------------
        # 10. HANDLE USER INPUT
        # --------------------------------------------------------
        if prompt := st.chat_input("Ask about NBA players, teams, or stats..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            answer = None
            used_mode = None

            # ------------------ 10.1 Live stats path ------------------
            live_used = False
            if looks_like_live_stats_question(prompt):
                player_name = extract_player_name_simple(prompt)
                if player_name:
                    # current season 
                    current_year = datetime.now().year
                    season = current_year - 1

                    try:
                        stats, err = get_player_season_average(player_name, season)
                        if err:
                            answer = err
                            used_mode = "Live API (error)"
                        else:
                            answer = format_season_average_human(stats)
                            used_mode = "Live API"
                        live_used = True
                    except LiveStatsError as e:
                        # Gracefully handle auth / network errors instead of crashing
                        answer = (
                            "Live NBA stats are currently unavailable.\n\n"
                            f"{e}\n\n"
                            "The app will still work using the CSV data (Pandas / RAG)."
                        )
                        used_mode = "Live API (unavailable)"
                        live_used = True
                # If we couldn't detect a player name, we just fall through to RAG/Pandas

            # ------------------ 10.2 Pandas / RAG routing ------------------
            if not live_used:
                # Auto mode routing
                auto_beginner = is_beginner_question(prompt)
                auto_use_rag = (mode == "Auto (Smart)") and auto_beginner
                auto_use_pandas = (mode == "Auto (Smart)") and not auto_beginner

                # Explicit modes
                force_pandas = (mode == "Pandas Agent")
                force_rag = (mode == "RAG")

                # -------- Try Pandas Agent (if chosen) --------
                if force_pandas or auto_use_pandas:
                    with st.spinner("Using Pandas Agent (dataframe reasoning)..."):
                        try:
                            response = pandas_agent.invoke(prompt)
                            answer = response["output"]
                            used_mode = "Pandas Agent"
                        except Exception:
                            st.warning(
                                "Pandas Agent had an issue. "
                                "Switching to RAG (semantic search + knowledge) if available."
                            )
                            # If Pandas fails, we fall back to RAG if possible
                            force_rag = True

                # -------- Try RAG (if chosen / fallback) --------
                if (force_rag or auto_use_rag) and rag_available and answer is None:
                    with st.spinner("Using RAG (semantic search + NBA knowledge)..."):
                        try:
                            response = rag_chain.invoke({"query": prompt})
                            answer = response["result"]
                            if used_mode == "Pandas Agent":
                                used_mode = "RAG (fallback from Pandas)"
                            elif mode == "Auto (Smart)" and auto_beginner:
                                used_mode = "RAG (Auto - beginner/concept question)"
                            else:
                                used_mode = "RAG"
                        except Exception as e:
                            st.error(f"RAG Error: {e}")

            # ------------------ 10.3 Display answer ------------------
            with st.chat_message("assistant"):
                if answer:
                    if used_mode is None:
                        badge = "🤖"
                        used_mode = "Unknown"
                    else:
                        if "Live API" in used_mode:
                            badge = "🌐"
                        elif "Pandas" in used_mode:
                            badge = "📊"
                        else:
                            badge = "🔍"
                    st.caption(f"{badge} Mode: {used_mode}")
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                else:
                    st.error("Failed to get an answer. Try rephrasing the question.")

    except FileNotFoundError:
        st.error("CSV files missing! Please run 'python get_data.py' first.")
else:
    st.info("Please enter your Groq API Key in the sidebar to start.")

