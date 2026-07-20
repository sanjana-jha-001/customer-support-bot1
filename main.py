import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


hf_token = st.secrets["HF_Token"]

st.title("🤖 Customer Support Chatbot")

# Keep conversation history across turns
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=(
            "You are a friendly, helpful customer support assistant. "
            "Be concise, polite, and clear. If you don't know something, "
            "say so honestly instead of making it up."
        ))
    ]

# Show past messages
for msg in st.session_state.messages[1:]:  # skip system message
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# Chat input
user_input = st.chat_input("Type your question here.. eg :- how to change password ? ")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=hf_token,
        provider="auto",
        temperature=0.3,
        max_new_tokens=512,
    ))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(st.session_state.messages)
            st.write(response.content)

    st.session_state.messages.append(AIMessage(content=response.content))
