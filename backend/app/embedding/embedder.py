from langchain_huggingface import HuggingFaceEmbeddings
from ..core.config import settings
from uuid import uuid4


model_name = settings.EMBEDDING_MODEL
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def create_embeddings(docs):
    """Create embeddings of the specified texts"""

    vectors = []
    for doc in docs:
        text = doc.page_content
        metadata = doc.metadata
        doc_id = doc.id or str(uuid4())      # Generate random id if id in docs is null

        embedding = hf.embed_query(text)

        vectors.append({
            "id": doc_id,
            "values": embedding,
            "metadata": metadata,
            }
        )
    
    return vectors





