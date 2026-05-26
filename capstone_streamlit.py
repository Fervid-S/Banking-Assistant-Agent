import streamlit as st
import uuid
import os
from agent import BankingAssistant
from langchain_core.messages import HumanMessage, AIMessage

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

st.set_page_config(page_title="SecureBank AI Portal", layout="wide")

def load_local_documents(directory="data"):
    documents = {}
    if not os.path.exists(directory):
        os.makedirs(directory)
        return documents
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                documents[filename] = f.read()
    return documents

@st.cache_resource
def get_assistant():
    assistant = BankingAssistant()
    kb_docs = load_local_documents("data")
    
    if not kb_docs:
        st.error("No documents found in the /data folder. Please upload your .txt files.")
        st.stop()
    assistant.ingest_kb(kb_docs)
    return assistant

engine = get_assistant()
graph = engine.compile_workflow()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Session Management")
    if st.button("Reset Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()
    st.caption(f"Active Thread: {st.session_state.thread_id}")

st.title("SecureBank FAQ Assistant")

# DISPLAY: Show existing history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        if chat["role"] == "assistant" and "metadata" in chat:
            meta = chat["metadata"]
            st.caption(f"Faithfulness: {meta['score']} | Sources: {meta['sources']}")

# LOGIC: Process NEW input only
if prompt := st.chat_input("Ask about loans, accounts, cards...."):
    # 1. Add user message to UI immediately
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Reconstruct proper LangGraph Message state history mapping
    formatted_messages = []
    for msg in st.session_state.chat_history[:-1]:
        if msg["role"] == "user":
            formatted_messages.append(HumanMessage(content=msg["content"]))
        else:
            formatted_messages.append(AIMessage(content=msg["content"]))
    
    # Append the newest current prompt
    formatted_messages.append(HumanMessage(content=prompt))

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    with st.spinner("Consulting Knowledge Base..."):
        result = graph.invoke(
            {"messages": formatted_messages}, 
            config=config
        )
    
    # 3. Extract final elements cleanly
    final_msg = result["messages"][-1].content
    score = result.get("faithfulness_score", 1.0)
    topics = ", ".join(result.get("source_documents", ["N/A"]))
    
    # 4. Append final dictionary with metadata to history
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": final_msg,
        "metadata": {"score": score, "sources": topics}
    })
    
    # 5. Force UI refresh cleanly
    st.rerun()