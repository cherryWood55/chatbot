from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
from model_utils import ChatModel

from rag_utils import RAGRetriever

app = FastAPI()
chatbot = ChatModel()
with open("data/faqs.json") as f:
    faq_data = json.load(f)

class ChatRequest(BaseModel):
    user_query: str

# @app.post("/chat")
# async def chat(req: ChatRequest):
#     query = req.user_query.lower()

#     # Simple keyword-based intent detection (replace with NLP later)
#     if "return" in query:
#         context = faq_data["return_policy"]
#     elif "order" in query or "track" in query:
#         context = faq_data["track_order"]
#     elif "store" in query or "hours" in query:
#         context = faq_data["store_hours"]
#     elif "size" in query or "available" in query:
#         context = faq_data["size_availability"]
#     else:
#         context = "I'm not sure, but I'll try to help you."

#     prompt = f"""You are a helpful customer support agent.
# Context: {context}
# Customer: {query}
# Agent:"""

#     answer = chatbot.generate_response(prompt)
#     return {"response": answer.split("Agent:")[-1].strip()}

# ...

retriever = RAGRetriever()

@app.post("/chat")
async def chat(req: ChatRequest):
    query = req.user_query

    # RAG: retrieve top-k relevant chunks
    retrieved_context = retriever.retrieve(query)
    context = "\n".join(retrieved_context)

    prompt = f"""You are a helpful customer support agent.

Use the following context to answer the customer's question:

{context}

Customer: {query}
Agent:"""

    answer = chatbot.generate_response(prompt)
    return {"response": answer.split("Agent:")[-1].strip()}
