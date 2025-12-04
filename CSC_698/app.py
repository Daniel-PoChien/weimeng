import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os

# 1. Page Setup
st.set_page_config(page_title="NBA AI Analyst", page_icon="🏀")
st.title("🏀 NBA Stats Agent")

# 2. Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Groq API Key:", type="password")
    st.markdown("[Get a Free Key Here](https://console.groq.com/keys)")

if api_key:
    # 3. Load Data (If files exist)
    try:
        df_teams = pd.read_csv("nba_teams.csv")
        df_rosters = pd.read_csv("nba_rosters.csv")
        df_scores = pd.read_csv("nba_scores.csv")
        
        # 4. Setup the "Instant" AI Model (Fixes Rate Limit Error)
        llm = ChatGroq(
            temperature=0, 
            model_name="llama-3.1-8b-instant",  # <--- CHANGED TO FASTER MODEL
            groq_api_key=api_key
        )

        # 5. Create the Smart Agent
        agent = create_pandas_dataframe_agent(
            llm, 
            [df_teams, df_rosters, df_scores], 
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )

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
                with st.spinner("Analyzing stats..."):
                    try:
                        response = agent.invoke(prompt)
                        answer = response['output']
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {e}")

    except FileNotFoundError:
        st.error(" CSV files not found! Please run 'python get_data.py' first.")
else:
    st.info("Please enter your API Key to start.")