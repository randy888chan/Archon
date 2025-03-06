from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from archon.archon_graph import agentic_flow
from langgraph.types import Command
from utils.utils import write_to_log
    
app = FastAPI()

class InvokeRequest(BaseModel):
    message: str
    thread_id: str
    is_first_message: bool = False
    config: Optional[Dict[str, Any]] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}    

@app.post("/invoke")
async def invoke_agent(request: InvokeRequest):
    """Process a message through the agentic flow and return the complete response.

    The agent streams the response but this API endpoint waits for the full output
    before returning so it's a synchronous operation for MCP.
    Another endpoint will be made later to fully stream the response from the API.
    
    Args:
        request: The InvokeRequest containing message and thread info
        
    Returns:
        dict: Contains the complete response from the agent
    """
    try:
        config = request.config or {
            "configurable": {
                "thread_id": request.thread_id
            }
        }

        # Use a list to collect response chunks more efficiently
        response_chunks: List[str] = []
        
        write_to_log(f"Starting processing for thread {request.thread_id}")
        
        if request.is_first_message:
            write_to_log(f"Processing first message for thread {request.thread_id}")
            async for msg in agentic_flow.astream(
                {"latest_user_message": request.message}, 
                config,
                stream_mode="custom"
            ):
                # Append each chunk to the list instead of concatenating strings
                response_chunks.append(str(msg))
        else:
            write_to_log(f"Processing continuation for thread {request.thread_id}")
            async for msg in agentic_flow.astream(
                Command(resume=request.message),
                config,
                stream_mode="custom"
            ):
                # Append each chunk to the list instead of concatenating strings
                response_chunks.append(str(msg))

        # Join all chunks into a single response only at the end
        response = "".join(response_chunks)
        
        # Log response size for debugging
        response_length = len(response)
        write_to_log(f"Response size for thread {request.thread_id}: {response_length} characters")
        
        # Log a preview of the response
        preview_length = min(100, response_length)
        write_to_log(f"Response preview: {response[:preview_length]}{'...' if response_length > preview_length else ''}")
        
        return {"response": response}
        
    except Exception as e:
        write_to_log(f"Error processing message for thread {request.thread_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
