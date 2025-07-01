from typing import TypedDict, Annotated, Sequence, Optional
from fastapi import HTTPException

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode

from ..agent.tools import VectorSearchTool
from langchain_groq import ChatGroq
from ..core.config import settings


# State schema
class AgentState(TypedDict):
    query: str
    context: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    error_msg: Optional[HTTPException]
    max_iterations: int
    current_iteration: int


retriever_tool = VectorSearchTool()
tools = [retriever_tool]
llm = ChatGroq(
                temperature=0.7,
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.LLM_MODEL,
            ).bind_tools(tools=tools)


def agent(state: AgentState) -> AgentState:
    """Agent node that decides which tool to call"""

    print("[GRAPH NODE] LLM generation Invoked")
    # if context:
    context = state.get("context", "")
    query = state["query"]
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if current_iteration >= max_iterations:
        print("Max iterations reached. Providing answer with available context.")
        final_message = AIMessage(
            content=f"Based on the available information: {context if context else 'No relevant documents found.'}"
        )
        return {
            "messages": state["messages"] + [final_message],
            "current_iteration": current_iteration + 1
            }

    system_message = SystemMessage(
        content="You are a helpful assistant that answers questions based on retrieved documents.\n"
                "You have access to the following tools:\n\n"
                "{tools}\n\n"
                "Use the VectorSearch tool to find relevant information before answering questions.\n"
                "Base your answers on the retrieved content.\n"
                "Be precise and concise."
                )
                
    human_message = HumanMessage(content=query)
    if not state["messages"]:
        state["messages"] = [system_message, human_message]
    try:
        response = llm.invoke(state["messages"])
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Using Tools: {[tc['name'] for tc in response.tool_calls]}")
        return {"messages": list(state["messages"]) + [human_message, response],
                "current_iteration": current_iteration + 1}        

    except Exception as e:
        print(f"Error in ask_agent: {e}")
        return {"error_msg": HTTPException(status_code=503, detail="LLM service temporarily unavailable. Please try again.")}
    

def should_continue(state: AgentState) -> AgentState:
    """Decide whether the agent should continue tool calling"""

    if state["messages"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        else:
            return "exit"

    

graph = StateGraph(AgentState)

graph.add_node("agent", agent)

retriever_tool_node = ToolNode(tools)
graph.add_node("tools", retriever_tool_node)


graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, 
                            {
                                "tools": "tools",
                                "exit": END
                            })
graph.add_edge("tools", "agent")


rag_app = graph.compile()