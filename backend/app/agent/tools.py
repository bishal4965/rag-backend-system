from langchain.tools import BaseTool
from pinecone import Pinecone
from ..core.config import settings
from ..embedding.embedder import hf
from pydantic import Field, PrivateAttr


class VectorSearchTool(BaseTool):
    """Tool for searching a vector db to retrieve semantically similar chunks"""
    name: str = "DocumentSearch"
    description: str =  (
        "Search for relevant documents and information based on a query. "
        "Use this tool when you need to find specific information or context "
        "to answer questions. Input should be a clear search query or question."
    )
    # query: str = Field()
    method: str = Field(default="cosine", description="The algorithm for semantic similarity search.")

    _client: Pinecone = PrivateAttr()

    def __init__(self, method: str = "cosine"):
        super().__init__()
        try:
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self._client = pc.Index("file-embeddings-dense")
            self.method = method
        except Exception as e:
            print(f"Error initializing Pinecone client: {e}")
            raise

    def _run(self, query: str) -> str:
        """Execute the vector search"""
        print(f"[TOOL CALL] VectorSearchTool was invoked with query: {query}")
        query_embed = hf.embed_query(query)

        results = self._client.query(
            # namespace=settings.PINECONE_NAMESPACE
            vector=query_embed,
            top_k=3,
            include_metadata=True,
            metric=self.method,
            include_values=False
        )

        matches = getattr(results, "matches", []) 
        if not matches:
            return "No matching content found."

        chunks = [m.metadata.get("text", "") for m in matches if "text" in m.metadata]
        if not chunks:
            return "No text metadata found in the results."

        print(f"Retrieving {len(chunks)} document chunks")
        return "\n\n".join(chunks)
    

    # async def _arun(self, query: str) -> str:
    #     """Async version of the tool"""
    #     return self._run(query)




