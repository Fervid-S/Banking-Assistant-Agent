import os
from typing import Annotated, TypedDict, List
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] 
    context: str
    retry_count: int
    faithfulness_score: float
    source_documents: List[str]

class BankingAssistant:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        self.vectorstore = None
        self.memory = MemorySaver()

    def ingest_kb(self, documents: dict):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        chunks = []
        for filename, content in documents.items():
            topic = filename.replace('_', ' ').replace('.txt', '').title()
            for chunk in text_splitter.split_text(content):
                chunks.append(Document(page_content=chunk, metadata={"topic": topic}))
        # Persisting in memory is faster and reliable for Streamlit caching
        self.vectorstore = Chroma.from_documents(chunks, self.embeddings)

    def memory_node(self, state: AgentState):
        # Safely preserve existing context or initialize if it doesn't exist
        return {
            "retry_count": state.get("retry_count", 0),
            "context": state.get("context", ""),
            "source_documents": state.get("source_documents", [])
        }
    
    def router_node(self, state: AgentState):
        query = state['messages'][-1].content.lower()
        banking_keywords = ["account", "loan", "card", "rate", "charge", "limit", "locker", "kyc", "atm"]
        if any(word in query for word in banking_keywords): 
            return "retrieve"
        if any(word in query for word in ["time", "calculate", "date"]): 
            return "tool"
        return "skip"

    def retrieval_node(self, state: AgentState):
        query = state['messages'][-1].content
        results = self.vectorstore.similarity_search(query, k=3)
        context = "\n".join([r.page_content for r in results])
        sources = list(set([r.metadata['topic'] for r in results]))
        return {"context": context, "source_documents": sources}

    def skip_retrieval_node(self, state: AgentState):
        return {"context": "General conversation logic. No KB context used.", "source_documents": ["N/A"]}

    def tool_node(self, state: AgentState):
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return {"context": f"System Status: Online. Current Date/Time: {now}.", "source_documents": ["System Tool"]}
        except Exception as e:
            return {"context": f"Tool Error: {str(e)}", "source_documents": ["System Error"]}

    def answer_node(self, state: AgentState):
        # Fallback to empty string if history length is less than 2
        history_messages = state['messages'][:-1]
        history_text = " ".join([m.content for m in history_messages]) if history_messages else "No prior history."
        current_query = state['messages'][-1].content
    
        system_prompt = f"""You are a Banking FAQ Assistant.
    
USER PROFILE FROM CONVERSATION: {history_text}
    
GROUNDING RULES:
1. Use ONLY the context below to answer. 
2. If the user's location (rural/urban) or account type was mentioned earlier in the PROFILE, you MUST use that to give a specific answer.
3. Do not ask the user for information they have already provided.

STRICT RULE: You may ONLY use the information provided in the 'CONTEXT' section below. 
If the answer is not explicitly stated in the context, you must say: 
'I apologize, but I do not have that specific information in my records. Please contact 1800-BANK-HELP.'
    
DO NOT use your internal knowledge about banking, interest rates, or general financial rules.
    
BANKING KNOWLEDGE CONTEXT:
{state['context']}
    
USER QUESTION: {current_query}
    
ASSISTANT RESPONSE:"""
    
        response = self.llm.invoke(system_prompt)
        return {"messages": [response]}
    
    def eval_node(self, state: AgentState):
        ans = state['messages'][-1].content
        ctx = state['context']
        
        # If skip node or tool node was used, skip harsh evaluation grading
        if "General conversation logic" in ctx or "System Status" in ctx:
            return {"faithfulness_score": 1.0, "retry_count": state.get('retry_count', 0) + 1}
            
        eval_q = f"On a scale of 0.0 to 1.0, how faithful is this answer to the context? Return ONLY the number.\nContext: {ctx}\nAnswer: {ans}"
        try:
            score_text = self.llm.invoke(eval_q).content.strip()
            score = float(score_text)
        except: 
            score = 0.8
        return {"faithfulness_score": score, "retry_count": state.get('retry_count', 0) + 1}

    def save_node(self, state: AgentState):
        return {
            "messages": state['messages'],
            "context": state['context'],
            "source_documents": state['source_documents'],
            "faithfulness_score": state.get('faithfulness_score', 1.0)
        }

    def compile_workflow(self):
        builder = StateGraph(AgentState)
        builder.add_node("memory", self.memory_node)
        builder.add_node("retrieve", self.retrieval_node)
        builder.add_node("skip", self.skip_retrieval_node)
        builder.add_node("tool", self.tool_node)
        builder.add_node("answer", self.answer_node)
        builder.add_node("evaluate", self.eval_node)
        builder.add_node("save", self.save_node)

        builder.set_entry_point("memory")
        builder.add_conditional_edges("memory", self.router_node, {"retrieve": "retrieve", "tool": "tool", "skip": "skip"})
        builder.add_edge("retrieve", "answer")
        builder.add_edge("tool", "answer")
        builder.add_edge("skip", "answer")
        builder.add_edge("answer", "evaluate")
        
        builder.add_conditional_edges(
            "evaluate", 
            lambda x: "retry" if x["faithfulness_score"] < 0.7 and x["retry_count"] < 3 else "end", 
            {"retry": "answer", "end": "save"}
        )
        builder.add_edge("save", END)
        return builder.compile(checkpointer=self.memory)