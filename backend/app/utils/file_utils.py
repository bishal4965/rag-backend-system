from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os


def extract_text(contents: bytes, filename: str) -> str:
    """Extract text from the file contents"""
    
    try:
        if filename.endswith('pdf'):
            with open("temp.pdf", "wb") as f:
                f.write(contents)
            
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            return "\n".join([doc for doc in docs])
        
        elif filename.endswith('txt'):
            with open("temp.txt", "wb") as f:
                f.write(contents)

            loader = TextLoader("temp.pdf")
            docs = loader.load()
            return "\n".join([doc for doc in docs])
        
    except Exception as e:
        return f"Failed to extract text from {filename}: {str(e)}"
    
    finally:
        temp_file_paths = ["temp.pdf", "temp.pdf"]
        for temp_file_path in temp_file_paths:
            if os.path.exists():
                os.unlink(temp_file_path)