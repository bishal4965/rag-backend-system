from langgraph.prebuilt import create_react_agent
from .tools import VectorSearchTool
from langchain_groq import ChatGroq
from ..core.config import settings
from fastapi import HTTPException
from langchain_core.messages import HumanMessage, SystemMessage


tools = [VectorSearchTool()]
model = ChatGroq(
                temperature=0.7,
                groq_api_key=settings.GROQ_API_KEY,
                model_name=settings.LLM_MODEL,
            )

agent_executor = create_react_agent(model, tools)

def ask_agent(query):

    system_message = SystemMessage(
        content="You are a helpful assistant that answers questions based on retrieved documents. "
                "Always use the DocumentSearch tool to find relevant information before answering questions. "
                "Base your answers on the retrieved content and cite the sources when possible."
    )

    human_message = HumanMessage(content=query)
    try:
        response = agent_executor.invoke({"messages": [system_message, human_message]})
        # print(response)
        return response["messages"][-1].content

    except Exception as e:
        print(f"Error in ask_agent: {e}")
        raise HTTPException(status_code=503, detail="LLM service temporarily unavailable. Please try again.")

    