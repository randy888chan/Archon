import asyncio
import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to Python path to import the modules being tested
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock all dependencies before imports
mock_streamlit = MagicMock()
mock_session_state = MagicMock()
mock_session_state.messages = []  # Add messages attribute to the mock
mock_streamlit.session_state = mock_session_state
sys.modules['streamlit'] = mock_streamlit

sys.modules['langgraph.types'] = MagicMock()
sys.modules['langgraph.types'].Command = MagicMock()
sys.modules['archon.archon_graph'] = MagicMock()
mock_agentic_flow = AsyncMock()
sys.modules['archon.archon_graph'].agentic_flow = mock_agentic_flow

# Mock nest_asyncio before import
mock_nest_asyncio = MagicMock()
sys.modules['nest_asyncio'] = mock_nest_asyncio

# Mock for agentic_flow
class MockAgentic:
    async def astream(self, *args, **kwargs):
        yield "Test"
        yield " response"
        yield " chunk"

# Empty response mock
class EmptyResponseMock:
    async def astream(self, *args, **kwargs):
        # Just return without yielding anything
        return
        yield  # This will never be reached

# Now we can safely import the module we want to test
from streamlit_pages.chat import chat_tab, chat_tab_wrapper, run_agent_with_streaming

# Define test class
class TestChat:
    
    @pytest.fixture(autouse=True)
    def setup(self):
        # Reset mocks
        mock_streamlit.reset_mock()
        mock_session_state.messages = [{"type": "human", "content": "Hello"}]  # Initialize with a message
        mock_nest_asyncio.reset_mock()
        yield
    
    @pytest.mark.asyncio
    async def test_run_agent_with_streaming(self):
        """Test that run_agent_with_streaming yields expected chunks"""
        # Setup
        mock_session_state.messages = [{"type": "human", "content": "Hello"}]
        
        # Replace the agentic_flow with our mock
        mock_agentic_flow.astream = MockAgentic().astream
        
        # First message case
        chunks = []
        async for chunk in run_agent_with_streaming("Build an agent"):
            chunks.append(chunk)
        
        assert "".join(chunks) == "Test response chunk"
        
        # Continue conversation case
        mock_session_state.messages = [
            {"type": "human", "content": "Hello"},
            {"type": "ai", "content": "Hi there"},
            {"type": "human", "content": "How are you?"}
        ]
        
        chunks = []
        async for chunk in run_agent_with_streaming("How are you?"):
            chunks.append(chunk)
        
        assert "".join(chunks) == "Test response chunk"
    
    @pytest.mark.asyncio
    async def test_streaming_with_timeouts(self):
        """Test that streaming handles timeouts properly"""
        # Setup
        mock_session_state.messages = [{"type": "human", "content": "Hello"}]
        
        # Create a slow mock
        class SlowMock:
            async def astream(self, *args, **kwargs):
                yield "Initial response"
                await asyncio.sleep(0.1)  # Reduce sleep time for testing
                yield " after delay"
                await asyncio.sleep(0.1)  # Reduce sleep time for testing
                yield " final chunk"
        
        # Replace the agentic_flow with our slow mock
        mock_agentic_flow.astream = SlowMock().astream
        
        # Use a timeout
        chunks = []
        try:
            # Set a timeout of 0.15 seconds to catch after first delay but before final chunk
            async with asyncio.timeout(0.15):
                async for chunk in run_agent_with_streaming("Test with timeout"):
                    chunks.append(chunk)
        except asyncio.TimeoutError:
            # We expect a timeout because our mock has delays
            pass
        
        # We should have received the first two chunks but not the final one
        assert "Initial response after delay" in "".join(chunks)
        assert " final chunk" not in "".join(chunks)
    
    @pytest.mark.asyncio
    async def test_empty_response_handling(self):
        """Test that empty responses are handled properly with fallback messages"""
        # Setup
        mock_session_state.messages = [
            {"type": "human", "content": "Hello"},
            {"type": "ai", "content": "Hi there"},
            {"type": "human", "content": "yes"}  # This should trigger empty response handling
        ]
        
        # Replace with a mock that yields nothing
        mock_agentic_flow.astream = EmptyResponseMock().astream
        
        # Capture the response
        chunks = []
        async for chunk in run_agent_with_streaming("yes"):
            chunks.append(chunk)
        
        # Check if we got a fallback message
        response = "".join(chunks)
        assert len(response) > 0
        assert "No response was generated" in response or "agent was unable to generate" in response
        
    def test_chat_tab_wrapper(self):
        """Test that chat_tab_wrapper properly handles event loops"""
        # Reset the mock_nest_asyncio for this test
        mock_nest_asyncio.reset_mock()
        
        # Mock the asyncio module for this test
        mock_asyncio = MagicMock()
        mock_loop = MagicMock()
        mock_asyncio.get_event_loop.return_value = mock_loop
        
        # Patch only what we need for this test
        with patch('streamlit_pages.chat.asyncio', mock_asyncio):
            # Call the wrapper
            chat_tab_wrapper()
            
            # Verify the mocks were called correctly
            mock_nest_asyncio.apply.assert_called_once()
            mock_asyncio.get_event_loop.assert_called_once()
            mock_loop.run_until_complete.assert_called_once() 