from utils.utils import get_env_var
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from archon.archon_graph import agentic_flow
from langgraph.types import Command
from utils.utils import write_to_log
from dotenv import load_dotenv
import sqlite3
import httpx

import logfire

app = FastAPI()

load_dotenv()

# configure logfire

logfire.configure(token=os.getenv('LOGFIRE_API_KEY'))
logfire.instrument_fastapi(app, capture_headers=True)
logfire.instrument_sqlite3()

# sqlite database URL
db_url = 'https://files.pydantic.run/pydantic_pypi.db'

# download the data and create the database
with logfire.span('preparing database'):
    with logfire.span('downloading data'):
        r = httpx.get(db_url)
        r.raise_for_status()

    with logfire.span('create database'):
        with open('pydantic_pypi.db', 'wb') as f:
            f.write(r.content)
        connection = sqlite3.connect('pydantic_pypi.db')


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
            "thread_id": request.thread_id,
            "recursion_limit": 300  # Add higher recursion limit to avoid GraphRecursionError
        }

        # Log the config details for debugging
        write_to_log(f"Using config for thread {request.thread_id}: {config}")
        print(f"Graph config: {config}")

        response = ""
        if request.is_first_message:
            write_to_log(
                f"Processing first message for thread {request.thread_id}")

            async for msg in agentic_flow.astream(input={"latest_user_message": request.message}, config=config, stream_mode="custom"):
                response += str(msg)
        else:
            write_to_log(
                f"Processing continuation for thread {request.thread_id}")
            async for msg in agentic_flow.astream(
                input=Command(resume=request.message),
                config=config,
                stream_mode="custom"
            ):
                response += str(msg)

        write_to_log(
            f"Final response for thread {request.thread_id}: {response}")
        return {"response": response}

    except Exception as e:
        print(
            f"Exception invoking Archon for thread {request.thread_id}: {str(e)}")
        write_to_log(
            f"Error processing message for thread {request.thread_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
