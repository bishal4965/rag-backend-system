from typing import TypedDict, Annotated, Sequence, Optional
from fastapi import HTTPException

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from ..agent.tools import VectorSearchTool, InterviewBookingTool
from langchain_groq import ChatGroq
from ..core.config import settings

from .cache import init_semantic_cache
from .memory import checkpointer
from ..db.session import get_booking_db


MAX_HISTORY = 10

# State schema
class AgentState(TypedDict):
    query: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    error_msg: Optional[HTTPException]
   


# Initialize semantic cache
# init_semantic_cache(distance_threshold=0.15, ttl=7200)

db = next(get_booking_db())

retriever_tool = VectorSearchTool()
booking_tool = InterviewBookingTool(db)
# booking_tool.description = (
#     "For booking interviews. Call me multiple times as you gather information. "
#     "I'll guide you through collecting: full_name, email, date, time. "
#     "Always call me with just the new information provided by the user."
#     "Once all the user information are collected then send mail by invoking me and return the appointment summary."
# )

booking_tool.description = (
    "MANDATORY for all interview booking steps. Call me for EVERY piece of booking information. "
    "I'll guide you through collecting: full_name, email, date, time. "
    "Call me immediately when user provides any of these details. "
    "Call me with ONLY the new information from the user's last message. "
    "I validate inputs and will return errors for invalid data. "
    "After full validation, I automatically book and email confirmation."
)

tools = [retriever_tool, booking_tool]
llm = ChatGroq(
                temperature=0.1,
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.LLM_MODEL,
            ).bind_tools(tools=tools)


def agent(state: AgentState) -> AgentState:
    """Agent node that decides which tool to call"""

    print("[GRAPH NODE] LLM generation Invoked")
    
    query = state["query"]
    messages = state.get("messages", [])

    tool_descriptions = "\n".join(
        f"- {tool.name}: {tool.description}" 
        for tool in tools
    )

    system_message = SystemMessage(
        content="You are a helpful assistant that answers questions based on retrieved documents.\n"
                "You have access to the following tools:\n\n"
                "{tool_descriptions}\n\n"
                "Use the DocumentSearch tool to find relevant information before answering questions.\n"
                "Base your answers on the retrieved content.\n"
                "Be precise and concise.\n"
                "Use the InterviewBooking tool to help user book an appointment for interview.\n\n"

                "CRITICAL INSTRUCTIONS FOR BOOKING:\n"
                "1. Use InterviewBooking tool for EVERY booking step\n"
                "2. If tool returns 'VALIDATION_ERROR: <field>_INVALID', IMMEDIATELY re-ask for SAME field\n"
                "3. Never proceed to next field until current field passes validation\n"
                "4. When all fields are validated, tool will confirm booking automatically\n"


                "Follow the following workflow strictly\n"
                "Booking workflow:\n"
                "a. Collect name → b. Collect email → c. Schedule date → d. Schedule time\n\n"

                
                )
    
        
        # "2. If tool returns 'VALIDATION_ERROR: <field>_INVALID', re-ask the user for SAME field by returning the tool message.\n"
    
        # "4. For dates, always request in YYYY-MM-DD format and verify it's in the future\n"
        # "5. For times, request in 24h format (HH:MM) or 12h format (e.g., 2:30pm)\n"
    
    # "CRITICAL BOOKING INSTRUCTIONS:\n"
    #             "1. Use ONLY the InterviewBooking tool for booking steps\n"
    #             "2. Follow this EXACT workflow:\n"
    #             "   a. Collect full name → b. Collect email → c. Schedule date → d. Schedule time\n"
    #             "3. For each step:\n"
    #             "   - Call InterviewBooking with JUST the new information\n"
    #             "   - If it returns a VALIDATION_ERROR, re-ask for SAME field\n"
    #             "   - If it returns a question, output it VERBATIM\n"
    #             "4. Date format: ALWAYS use YYYY-MM-DD\n"
    #             "5. Time format: Use 12h (2:30pm) or 24h (14:30)\n"
    #             "6. After booking completion, the tool returns confirmation - show this to user\n\n"
                
    #             "NEVER skip steps or ask for multiple fields at once!"
    #             "Decide which tool to use based on user's intent."
    


    # "EXAMPLE WORKFLOW:\n"
    #             "- User: 'I want to book an interview'\n"
    #             "- You: 'Please provide your full name' (via InterviewBooking)\n"
    #             "- User: 'John Doe'\n"
    #             "- You: 'What's your email address?' (via InterviewBooking)\n"
    #             "- ... continues until all fields collected\n\n"
    
    # "Guide users through:\n"
    # "1. Collect name\n2. Collect email\n3. Schedule date\n4. Schedule time\n"
    # "Use InterviewBooking tool for each step - call it with new information as users provide it.\n"
    # "Never ask for multiple fields at once. Only ask for the next missing field.\n"
                
    human_message = HumanMessage(content=query)
    if not messages:
        messages = [system_message, human_message]
    
    elif messages and not isinstance(messages[-1], ToolMessage):
        # Adds human message to the maessage list if the last message is not a ToolMessage
        messages = messages + [human_message]
    try:
        trimmed_msg = [messages[0]] + messages[-(MAX_HISTORY - 1):]
        response = llm.invoke(trimmed_msg)
        print(response)
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"Using Tools: {[tc['name'] for tc in response.tool_calls]}")
        trimmed_msg = trimmed_msg + [response]
        
        return {"messages": trimmed_msg,
                "query": query,
                "error_msg": None
            }        

    except Exception as e:
        print(f"Error in ask_agent: {e}")
        return {"error_msg": HTTPException(status_code=503, detail="LLM service temporarily unavailable. Please try again.")}
    

def should_continue(state: AgentState) -> AgentState:
    """Decide whether the agent should continue tool calling"""

    for msg in reversed(state["messages"]):
        if hasattr(msg, "tool_calls"):
            return "tools" if msg.tool_calls else "exit"
    
    return "exit"


    

graph = StateGraph(AgentState)

graph.add_node("agent", agent)

tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)


graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, 
                            {
                                "tools": "tools",
                                "exit": END
                            })
graph.add_edge("tools", "agent")


rag_app = graph.compile(checkpointer=checkpointer)