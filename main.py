from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from service import rag_chain, initialize_pinecone_index, create_embeddings
from langchain_core.messages import HumanMessage, AIMessage
import os

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    history: list  

@app.on_event("startup")
async def startup_event():
    print("Initializing services...")
    initialize_pinecone_index()
    if not os.getenv("SKIP_EMBEDDINGS", False):
        create_embeddings()
    print("Service ready!")

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        # Convert history format
        converted_history = []
        for msg in chat_request.history:
            if msg["type"] == "human":
                converted_history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                converted_history.append(AIMessage(content=msg["content"]))
        
        # Process the message
        result = rag_chain.invoke({
            "input": chat_request.message,
            "chat_history": converted_history
        })
        
        return {
            "answer": result["answer"],
            "history": [
                {"type": "human", "content": chat_request.message},
                {"type": "ai", "content": result["answer"]}
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "running"}