import streamlit as st
from openai import OpenAI
import requests
import json
import time

# Show title and description.
st.title("💬 Ragooon Chatbot")
st.write(
    "API for RagoonBot, a custom LLM model version 0.1"
)
    
# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What can I help you today?"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    url = 'https://ragoooon.onrender.com/stream_complete'
    myobj = {"prompt": prompt,"history": []}
    stream = requests.post(url, json = myobj)
    
    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    
    def stream_data():
        for word in stream.json()['stream']:
            yield word
            time.sleep(0.02)
    
    with st.chat_message("assistant"):
        response = st.write_stream(stream_data)
    st.session_state.messages.append({"role": "assistant", "content": response})