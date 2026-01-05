import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
# DuckDuckGoSearchRun-->gives capability to search from the internet
from langchain_classic.agents import initialize_agent,AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
# """
# What does StreamlitCallbackHandler do?

# It:

# Streams tokens as they’re generated

# Shows agent thoughts

# Displays tool inputs / outputs

# Updates Streamlit UI in real time

# Instead of waiting for the full response, users see the agent “thinking”.
# """
from dotenv import load_dotenv
load_dotenv()

# Arxiv and Wikipedia Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

search=DuckDuckGoSearchRun(name="Search")

st.title("LangChain- Chat with search")
"""
In this example, we're using StreamlitCallbackHandler to display the thoughts and actions
of the agent when they are executing or interacting with the tools
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API key:",type="password")

if "messages" not in st.session_state:
    st.session_state['messages']=[
        {"role":"assistant","content":"Hi I'm a chatbot who can search the web. How can I help you"}

    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content']) # We are writing them in key value pair


if prompt:=st.chat_input(placeholder="What is machine Learning"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    # Whenever we write any prompt it is going to be appended in this particular messages
    llm=ChatGroq(groq_api_key=api_key,model="llama-3.1-8b-instant",streaming=True)
    tools=[arxiv,wiki,search]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    # """
    # Diff b/w them is how they generate prompt for the language model
    # ZERO_SHOT_REACT_DESCRIPTION-->Don't rely on the chat history. Make decision on the 
    # current input only without  relying on any prev actions
    # CHAT_ZERO_SHOT_REACT_DESCRIPTION-->Uses a chat history to remember the context of the chat
    # and the history of the conversation and accept a certain structure in the chat history
    # """
#  handling_parsing_errors=True --> If you get any erros make sure to parse them

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False) #Whenever my agent is communicating with itself . We  should be
        # able to see that
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":'assistant',"content":response})
        st.write(response)

