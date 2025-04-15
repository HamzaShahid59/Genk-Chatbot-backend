from fastapi import FastAPI, HTTPException , WebSocket
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

@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()  # <-- This was missing
    try:
        # Receive client message as JSON
        data = await websocket.receive_json()
        
        # Parse data into ChatRequest
        chat_request = ChatRequest(**data)
        
        # Rest of your original processing logic
        converted_history = []
        for msg in chat_request.history:
            if msg["type"] == "human":
                converted_history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                converted_history.append(AIMessage(content=msg["content"]))
        
        output = {}
        full_answer = ""
        async for chunk in rag_chain.astream({
            "input": chat_request.message,
            "chat_history": converted_history
        }):
            for key in chunk:
                if key not in output:
                    output[key] = chunk[key]
                else:
                    output[key] += chunk[key]

                # Stream each token to the WebSocket
                if key == "answer":
                    new_text = chunk[key].replace(full_answer, "")
                    if new_text:
                        full_answer += new_text
                        await websocket.send_json({
                            "type": "chunk",
                            "content": new_text
                        })
        
        
        await websocket.send_json({
            "type": "complete",
            "answer": full_answer,
            "chat_history": [
                {"type": "human", "content": chat_request.message},
                {"type": "ai", "content": full_answer}
            ]
        })
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

@app.get("/")
def health_check():
    return {"message": "API is Running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "running"}