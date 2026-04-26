import streamlit as st
import uuid
import os
from agent import BankingAssistant
from langchain_core.messages import HumanMessage

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

# LOGIC: Process NEW input only
if prompt := st.chat_input("Ask abou loans, accounts, cards...."):
    # 1. Add user message to UI immediately
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. RUN GRAPH (Pass only the current human message)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # Use a spinner so the user knows it's thinking
    with st.spinner("Consulting Knowledge Base..."):
        result = graph.invoke(
            {"messages": [HumanMessage(content=prompt)]}, 
            config=config
        )
    
    # 3. EXTRACT: Only grab the very last message from the graph
    # This prevents the 'double answer' if the graph retried internally
    final_msg = result["messages"][-1].content
    
    # 4. APPEND & DISPLAY ASSISTANT
    with st.chat_message("assistant"):
        st.markdown(final_msg)
        st.caption(f"Sources: {', '.join(result.get('source_documents', ['N/A']))}")
    
    st.session_state.chat_history.append({"role": "assistant", "content": final_msg})
    
    # 5. FORCE STOP: Prevents Streamlit from looping back and rerunning logic
    st.rerun()

    response = result["messages"][-1].content
    score = result.get("faithfulness_score", 0.0)
    topics = ", ".join(result.get("source_documents", ["N/A"]))

    with st.chat_message("assistant"):
        st.markdown(response)
        st.divider()
        st.caption(f"Faithfulness: {score} | Sources: {topics}")
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})