# RAG Agent with Interview Booking API



> FastAPI-based conversational agent with document retrieval and interview booking capabilities

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-FF6F00?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Pinecone](https://img.shields.io/badge/Pinecone-430098?style=flat)](https://www.pinecone.io/)

## Features

- **Conversational RAG Agent** with document retrieval
- **Interview Booking System** with step-by-step validation
- **Stateful Workflow** using LangGraph and Redis for conversation management
- **Email Confirmation** for booked interviews


## Installation

### Prerequisites
- Python 3.10+
- pip
- Pinecone API key
- Groq API key
- SMTP credentials for email sending

### Setup
1. Clone repository:
```bash
git clone https://github.com/bishal4965/rag-backend-system.git
cd rag-backend-system
```

2. Create virtual environment:
```bash
python -m venv .venv
```

3. Install dependencies
```bash
pip install -r requirements.txt
```


