#!/usr/bin/env python
"""
Test script to verify LangGraph configuration fixes.
Run this script to test if the 'unexpected keyword argument config' error is fixed.
"""

from utils.utils import write_to_log
from archon.archon_graph import agentic_flow
import asyncio
import sys
import os
from langgraph.types import Command

# Add the project root to the path so we can import from archon
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


async def test_invoke_with_config():
    """Test invoking the graph with a config dictionary"""
    print("\n--- Testing graph invoke with config ---")

    # Use the standard LangGraph config structure
    config = {
        "configurable": {
            "thread_id": "test-thread-1"
        },
        "recursion_limit": 300  # Keep recursion limit high for invoke tests
    }

    print(f"Using config: {config}")

    try:
        # Test a simple invocation with the config
        result = await agentic_flow.ainvoke(
            {"latest_user_message": "Test message with config"},
            config=config
        )
        print("Result: Success! No errors with config parameter.")
        print(f"Result keys: {result.keys() if result else 'No result'}")
        return True
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return False


async def test_stream_with_config():
    """Test streaming the graph with a config dictionary"""
    print("\n--- Testing graph streaming with config ---")

    # Use the standard LangGraph config structure
    config = {
        "configurable": {
            "thread_id": "test-thread-2"
        },
        "recursion_limit": 100  # Use appropriate limit for streaming tests
    }

    print(f"Using config: {config}")

    try:
        # Just test if it starts streaming without errors
        print("Starting streaming...")
        count = 0
        async for chunk in agentic_flow.astream(
            {"latest_user_message": "Build me an agent that can search and summarize academic papers"},
            config,
            stream_mode="values"
        ):
            count += 1
            if count <= 3:  # Only print first few chunks to avoid flooding output
                print(f"Received chunk {count}: {type(chunk)}")

            # Break after a few chunks to keep the test short
            if count >= 5:
                print("Successfully received 5 chunks, stopping stream.")
                break

        print("Result: Success! No errors with config parameter in streaming.")
        return True
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return False


async def run_tests():
    """Run all tests and return overall success status"""
    print("=== LangGraph Configuration Tests ===")
    print("Testing if configuration parameters are properly handled")

    results = []

    # Test 1: Basic invocation
    results.append(await test_invoke_with_config())

    # Test 2: Streaming
    results.append(await test_stream_with_config())

    # Summary
    success_count = sum(1 for r in results if r)
    print("\n=== Test Results ===")
    print(f"Passed: {success_count}/{len(results)} tests")

    if all(results):
        print("\n✅ All tests PASSED! The configuration issue appears to be fixed.")
        return 0
    else:
        print("\n❌ Some tests FAILED. The configuration issue may not be completely fixed.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)
