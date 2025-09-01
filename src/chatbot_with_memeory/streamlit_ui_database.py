import streamlit as st
from chatbot import ChatBot
from workflow import retrieval_all_thread
from langchain_core.messages import AIMessage, HumanMessage
import uuid

# -----------------------------
# Utils
# -----------------------------
def generate_thread_id():
    return str(uuid.uuid4())[:8]

def load_conversation(thread_id):
    state = ChatBot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

def reset_thread():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["messages"] = []
    add_thread(thread_id)

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# -----------------------------
# Session Setup
# -----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieval_all_thread() 

add_thread(st.session_state["thread_id"])
# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")

# Sidebar
st.sidebar.title("RAG Chatbot")
st.sidebar.button("➕ New Chat", on_click=reset_thread)
st.sidebar.header("💬 My Conversations")

for thread in st.session_state["chat_threads"][::-1]:  # latest first
    if st.sidebar.button(f"🗂️ {thread}"):
        st.session_state["thread_id"] = thread
        st.session_state["messages"] = load_conversation(thread)

        temp_messages = []
        for msg in st.session_state["messages"]:
            if isinstance(msg, HumanMessage):
                temp_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                temp_messages.append({"role": "assistant", "content": msg.content})
        st.session_state["messages"] = temp_messages

# -----------------------------
# Chat Interface
# -----------------------------
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response_container = st.chat_message("assistant")
    with response_container:
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🤔 *AI is thinking...*")

    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    full_response = ""

    with response_container:
        for message_chunk, metadata in ChatBot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            content = message_chunk.content
            full_response += content
            thinking_placeholder.markdown(full_response)

    st.session_state["messages"].append(
        {"role": "assistant", "content": full_response}
    )
