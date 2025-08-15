#!/usr/bin/env python3
"""
Fix missing EMBEDDING_PROVIDER setting in the database.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python', 'src'))

from server.services.credential_service import credential_service

async def fix_embedding_provider():
    """Add the missing EMBEDDING_PROVIDER setting."""
    try:
        # Check if EMBEDDING_PROVIDER already exists
        existing = await credential_service.get_credential("EMBEDDING_PROVIDER")
        if existing:
            print(f"EMBEDDING_PROVIDER already exists: {existing}")
            return
        
        # Add the missing EMBEDDING_PROVIDER setting
        success = await credential_service.set_credential(
            key="EMBEDDING_PROVIDER",
            value="openai",
            category="rag_strategy",
            description="Embedding provider to use: openai, ollama, or google"
        )
        
        if success:
            print("✅ Successfully added EMBEDDING_PROVIDER setting")
        else:
            print("❌ Failed to add EMBEDDING_PROVIDER setting")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(fix_embedding_provider())
