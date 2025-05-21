import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

load_dotenv()

# initilaze Tools
ArxivWrapper=ArxivAPIWrapper(top_k_results=1,doc_conten_char_max=300)
arxivtool=ArxivQueryRun(api_wrapper=ArxivWrapper)

Wikipediawrapper=WikipediaAPIWrapper(top_k_results=1,doc_conten_char_max=300)
wiki_tool=WikipediaQueryRun(api_wrapper=Wikipediawrapper)

search_tool=DuckDuckGoSearchRun(name="Search")

# streamlit ui

#st.set_page_config(page_title="End-to-End Search Engine Gen AI App.")
st.title("End-to-End Search Engine Gen AI App.")
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Please Enter your api key",type="password")

# chat_history ---- first_visit

if "messages" not in st.session_state:
    st.session_state.messages=[
        {
            "role":"assistant",
            "content":"Hi I am Chat Bot who can search the web. How can i help you?"
        }
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# New User Prompt
if prompt := st.chat_input("What is...?"):
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(
        api_key=api_key,
        model_name="Llama3-8b-8192",
        streaming=True,
    )
    tools=[search_tool,arxivtool,wiki_tool]
    # Create an Agent that uses ZERO REACT REACTION
    agent=initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parser_errors=True
    )
    # get and diplay the response
    with st.chat_message("assistant"):
        cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=agent.run(prompt,callbacks=[cb])
        st.session_state.messages.append({
            "role":"assistant","content":response
        })
        st.write(response)
