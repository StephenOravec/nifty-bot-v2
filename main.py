from fastapi import FastAPI, Request, HTTPException
import os
from openai import OpenAI
from agents import Agent, Runner
from google.cloud import firestore

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
db = firestore.Client()
app = FastAPI()

agent = Agent(
    name="Nifty-Bot",
    instructions=(
        "You are nifty-bot, a friendly AI agent inspired by the White Rabbit from Alice in Wonderland. "
        "You adore rabbit-themed NFTs on Ethereum L1 and L2. You often worry about the time. "
        "Be short, conversational, and rabbit-themed."
    ),
    model="gpt-4o-mini",
)

runner = Runner()


# ----------------------
# Memory Helpers
# ----------------------

def get_memory(user_id: str, limit: int = 20):
    """Retrieve last `limit` messages from Firestore."""
    doc_ref = db.collection("niftybotv2").document(user_id)
    doc = doc_ref.get()

    if doc.exists:
        messages = doc.to_dict().get("messages", [])
        return messages[-limit:]

    return []


def save_message(user_id: str, role: str, text: str):
    """Append new chat messages to Firestore."""
    doc_ref = db.collection("niftybotv2").document(user_id)
    doc_ref.set(
        {
            "messages": firestore.ArrayUnion([{"role": role, "text": text}])
        },
        merge=True
    )


# ----------------------
# Agents SDK Response Helper
# ----------------------

async def run_agent_with_memory(user_id: str, user_message: str):
    """
    Runs the OpenAI Agent with Firestore memory stitched in.
    """

    # Pull recent messages
    memory = get_memory(user_id)

    # Convert memory to the format expected by the Agents SDK
    contextual_messages = []
    for m in memory:
        contextual_messages.append({
            "role": m["role"],
            "content": m["text"]
        })

    # Add the new user message
    contextual_messages.append({
        "role": "user",
        "content": user_message
    })

    # Run agent with memory included as input
    result = runner.run(
        agent,
        input=contextual_messages
    )

    # Extract assistant text
    # (Agents SDK: result.output_text is always the final string)
    reply = result.output_text

    return reply


# ----------------------
# Routes
# ----------------------

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    message = data.get("message", "").strip()

    if not user_id or not message:
        raise HTTPException(status_code=400, detail="user_id and message required")

    # Run agent with memory
    reply = await run_agent_with_memory(user_id, message)

    # Save both user and assistant messages
    save_message(user_id, "user", message)
    save_message(user_id, "assistant", reply)

    return {"response": reply}


@app.get("/health")
def health_check():
    return {"status": "ok"}