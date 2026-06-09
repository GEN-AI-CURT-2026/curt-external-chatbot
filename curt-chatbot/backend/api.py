import os
import uvicorn
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline import CURTRagPipeline
from memory_manager import MemoryManager

app = FastAPI(
    title="CURT AI Chatbot Production API",
    description="Asynchronous API engine powered by Stage 3 & 4 Reliability Protocols.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline = CURTRagPipeline()
memory_manager = MemoryManager(window_size=5)


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

class SourceDocumentResponse(BaseModel):
    source: str
    content: str
    score: Optional[float] = None

class ChatResponse(BaseModel):
    """Aligned outbound data model with follow-ups completely removed."""
    answer: str
    status: str
    confidence_score: float
    sources: List[SourceDocumentResponse] = []


@app.post("/api/chat", response_model=ChatResponse, tags=["AI Chat Execution"])
async def chat_endpoint(request: ChatRequest):
    """Processes incoming messages using strong error insulation boundaries."""
    try:
        raw_history = [{"role": msg.role, "content": msg.content} for msg in request.history]
        managed_history = memory_manager.get_recent_history(raw_history)
        
        loop = asyncio.get_running_loop()
        
        pipeline_output = await loop.run_in_executor(
            None, rag_pipeline.run, request.message, managed_history
        )
        
        formatted_sources = []
        for doc in pipeline_output.get("sources", []):
            formatted_sources.append(
                SourceDocumentResponse(
                    source=os.path.basename(doc.metadata.get("source", "Unknown")),
                    content=doc.page_content,
                    score=doc.metadata.get("raw_score")
                )
            )
            
        return ChatResponse(
            answer=pipeline_output.get("answer"),
            status=pipeline_output.get("status"),
            confidence_score=pipeline_output.get("confidence_score", 0.0),
            sources=formatted_sources
        )
        
    except Exception as e:
        print(f"API Invocation Breakdown: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An issue occurred within the reliability engine layers."
        )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)