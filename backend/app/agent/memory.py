from redis import Redis
from langgraph.checkpoint.redis import RedisSaver

from ..core.config import settings


redis_client = Redis.from_url(settings.REDIS_MEMORY_URL)

# LangGraph state persistence
checkpointer = RedisSaver(redis_client=redis_client)        # , ttl=3600
checkpointer.setup()        # ensures Redis is reachable and initialized
