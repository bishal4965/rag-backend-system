import os
from langchain_redis.cache import RedisSemanticCache
from langchain_core.globals import set_llm_cache

from ..embedding.embedder import hf
from ..core.config import settings


def init_semantic_cache(distance_threshold: float = 0.2, ttl: int = 3600):
    """Initialize Redis based semantic cache for LLM calls"""

    semantic_cache = RedisSemanticCache(
        embeddings=hf,
        redis_url=settings.REDIS_CACHE_URL,
        distance_threshold=distance_threshold,
        ttl=ttl
    )

    # Register semantic cache globally
    set_llm_cache(semantic_cache)