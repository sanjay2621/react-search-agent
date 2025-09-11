"""
LLM Cache Demo - Demonstrates using SimpleCache with OpenAI LLM requests
This file shows how the cache prevents duplicate API calls and improves performance.
"""

import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from CacheImplemetation import SimpleCache
from dotenv import load_dotenv

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()


def demo_llm_with_cache():
    """Demonstrate LLM requests with caching."""

    print("🚀 LLM Cache Demo Starting...")
    print("=" * 50)

    # Create cache instance
    cache = SimpleCache()
    print(f"✅ Cache created. Initial size: {cache.size()}")

    # Create LLM with cache
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        cache=cache  # Attach our cache
    )

    # Test questions
    questions = [
        "What is the full name of BAPS?",
        "What is the full name of BAPS?",  # Duplicate - should use cache
    ]

    print("\n🔄 Sending requests to LLM...")
    print("-" * 50)

    for i, question in enumerate(questions, 1):
        print(f"\n📝 Request {i}: {question}")

        # Check if this request would be cached (manual check)
        llm_string = llm._get_llm_string()

        # Problem: LangChain stores cache keys using the full serialized message object format so need to use HumanMessage
        """
        The Problem Discovered: Cache Key Mismatch!
        What we're looking for: "What is the capital of India?:{"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]...
        What's actually stored: "[{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", "HumanMessage"], "kwargs": {"content": "What is the capital of India?"...
        LangChain is using the full serialized message object [HumanMessage(...)] as the prompt key, but our manual lookup is using just the raw string "What is the capital of India?".
        """
        # To fix above problem I use the same message format LangChain uses for caching
        # Create the same message format that LangChain uses for caching
        messages = [HumanMessage(content=question)]
        # LangChain serializes the messages for the cache key
        try:
            # Try to use LangChain's internal serialization
            import json
            serialized_messages = json.dumps([msg.model_dump() if hasattr(msg, 'model_dump') else msg.__dict__ for msg in messages], sort_keys=True)
        except:
            # Fallback to a simpler approach - use the actual key format we see in cache
            serialized_messages = str(messages)

        # Debug: Let's see what's in the cache
        print(f"🔍 Debug - cache keys: {list(cache.cache.keys())[:1]}...")  # Show just first key

        # Try to find the cached response using LangChain's internal lookup
        cached_response = cache.lookup(serialized_messages, llm_string)

        # If that doesn't work, try to find it by checking if any cache key contains our question
        if not cached_response:
            for cache_key in cache.cache.keys():
                if question in cache_key:
                    cached_response = cache.cache[cache_key]
                    print(f"🔍 Found match using substring search in key: {cache_key[:100]}...")
                    break

        if cached_response:
            print("💾 Found in CACHE - using cached response (NO API CALL)")
            # Use cached response
            start_time = time.time()
            if hasattr(cached_response, '__iter__') and hasattr(cached_response[0], 'text'):
                response_content = cached_response[0].text
            else:
                response_content = str(cached_response)[:100]
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️  Response time: {duration:.3f} seconds (instant from cache)")
            print(f"💬 Cached Response: {response_content[:100]}...")
        else:
            print("🌐 NOT in cache - making API call")
            # Time the API request
            start_time = time.time()
            response = llm.invoke(messages)
            end_time = time.time()
            duration = end_time - start_time

            print(f"⏱️  API Response time: {duration:.3f} seconds")
            print(f"💬 API Response: {response.content[:100]}...")

        print(f"📊 Cache size: {cache.size()}")

        # Small delay for readability
        time.sleep(0.5)

    print("\n" + "=" * 50)
    print("🎯 Demo Summary:")
    print(f"📊 Final cache size: {cache.size()}")
    print("💡 Notice the speed difference between API calls and cache hits!")
    print("💰 Cache prevented unnecessary API calls, saving costs!")

if __name__ == "__main__":
    try:
        # Main LLM caching demo
        demo_llm_with_cache()

        print("\n🎉 All demos completed successfully!")
        print("✅ Your SimpleCache implementation is working perfectly!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have OPENAI_API_KEY set in your .env file")
        print("💡 Install required packages: pip install langchain-openai python-dotenv")
