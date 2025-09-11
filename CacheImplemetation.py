## can you implement a simple in-memory cache in Python with methods to set, get, and delete items, as well as a method to clear the entire cache?
from typing import Any, Optional, Sequence
from langchain_core.outputs import Generation
# can we use BaseCache from cachetools?
#     from cachetools import Cache as BaseCache
#     class SimpleCache(BaseCache):
from langchain_core.caches import BaseCache


class SimpleCache (BaseCache):
        def __init__(self):
            self.cache = {}

        # Required BaseCache methods for LangChain compatibility
        def lookup(self, prompt: str, llm_string: str) -> Optional[Sequence[Generation]]:
            """Look up based on prompt and llm_string."""
            key = self._get_cache_key(prompt, llm_string)
            cached_result = self.cache.get(key)
            if cached_result:
                return cached_result
            return None

        def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
            """Update cache based on prompt and llm_string."""
            key = self._get_cache_key(prompt, llm_string)
            self.cache[key] = return_val

        def clear(self, **kwargs: Any) -> None:
            """Clear the entire cache."""
            self.cache.clear()

        def _get_cache_key(self, prompt: str, llm_string: str) -> str:
            """Generate a cache key from prompt and llm_string."""
            return f"{prompt}:{llm_string}"

        # Additional utility methods for manual cache management
        def set(self, key, value):
            """Set a value in the cache."""
            self.cache[key] = value

        def get(self, key):
            """Get a value from the cache. Returns None if the key does not exist."""
            return self.cache.get(key, None)

        def delete(self, key):
            """Delete a value from the cache. Does nothing if the key does not exist."""
            if key in self.cache:
                del self.cache[key]

        def size(self) -> int:
            """Return the number of items in the cache."""
            return len(self.cache)

        def contains(self, prompt: str, llm_string: str) -> bool:
            """Check if a prompt/llm_string combination exists in cache."""
            key = self._get_cache_key(prompt, llm_string)
            return key in self.cache
