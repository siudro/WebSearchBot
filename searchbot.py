import os
import nest_asyncio
import logging
import openai
import streamlit as st
from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo

# Set up logger
logger = logging.getLogger(__name__)

# Function to set up the Assistant
def setup_assistant(api_key: str) -> Assistant:
    openai.api_key = api_key
    llm_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
    llm = OpenAIChat(model=llm_model, api_key=api_key)
    return Assistant(
        name="web_search_assistant",
        llm=llm,
        description="You are a helpful Assistant called 'web search bud' and your goal is to assist the user in the best way possible.",
        instructions=[
            "If you don't find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If you need to reference the chat history, use the `get_chat_history` tool.",
            "If the user's question is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        ],
        show_tool_calls=False,
        search_knowledge=False,
        read_chat_history=True,
        tools=[DuckDuckGo()],
        markdown=True,
        add_chat_history_to_messages=True,
        add_datetime_to_instructions=True,
        debug_mode=True,
    )

# Function to run a query with the Assistant
def query_assistant(assistant: Assistant, question: str):
    response = ""
    for delta in assistant.run(question):
        response += delta 
    filtered_response = "\n".join(
        line for line in response.split("\n")
        if not line.startswith("Running:") and not line.startswith(question) and line.strip()
    )
    return filtered_response.strip()

# Streamlit UI
st.title("Chat with the whole internet🌐")

openai_access_token = st.text_input("OpenAI API Key", type="password")

if openai_access_token:
    st.subheader("Ask me anything and I will search the web for you!")

    # Initialize Assistant
    nest_asyncio.apply()
    assistant = setup_assistant(openai_access_token)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    user_input = st.text_input("Enter your question:")
    
    if user_input:
        #st.session_state.messages.append({"role": "user", "content": user_input})
        response = query_assistant(assistant, user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})

    for msg in st.session_state["messages"]:
        st.write(f"{msg['content']}")

else:
    st.write("Please enter your OpenAI API key to start chatting.")
