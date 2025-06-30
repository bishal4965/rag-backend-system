from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict

from ..agent.tools import VectorSearchTool
from langchain_groq import ChatGroq
from ..core.config import settings


# State schema
class AgentState(TypedDict):
    query: str
    context: str
    messages: list


retriever_tool = VectorSearchTool()
llm = ChatGroq(
                temperature=0.7,
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.LLM_MODEL,
            )

# NODES
def retrieve(state: AgentState) -> AgentState:
    print("[GRAPH NODE] Retrieval Invoked")
    context = retriever_tool._run(state["query"])
    return {"context": context}


def generate(state: AgentState) -> AgentState:
    print("[GRAPH NODE] LLM generation Invoked")

    context = state["context"]
    query = state["query"]

    prompt = f"""Use the following context to answer the question:

                Context:
                {context}

                Question:
                {query}
                """
