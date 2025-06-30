from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List
from ..core.config import settings


def extract_text(contents: bytes, filename: str) -> str:
    """Extract text from the file contents"""
    
    try:
        if filename.endswith('pdf'):
            with open("temp.pdf", "wb") as f:
                f.write(contents)
            
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            # return "\n".join([doc for doc in docs])
            return docs
        
        elif filename.endswith('txt'):
            with open("temp.txt", "wb") as f:
                f.write(contents)

            loader = TextLoader("temp.pdf")
            docs = loader.load()
            # return "\n".join([doc for doc in docs])
            return docs
        
    except Exception as e:
        return f"Failed to extract text from {filename}: {str(e)}"
    
    finally:
        temp_file_paths = ["temp.pdf", "temp.pdf"]
        for temp_file_path in temp_file_paths:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


def chunk_text(text: List[str], method: str = settings.CHUNKING_METHOD) -> List[str]:
    """Chunk text according to the specified 'method' strategy"""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(text)

    return texts




