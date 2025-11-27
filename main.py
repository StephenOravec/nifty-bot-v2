from fastapi import FastAPI, Request, HTTPException
import os
import sqlite3
import secrets
from openai import OpenAI
from agents import Agent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# ----------------------
# Agent Setup
# ----------------------
agent = Agent(
    name="Nifty-Bot",
    instructions=(
        "You are nifty-bot, a friendly AI agent inspired by the White Rabbit from "
        "Alice in Wonderland. You adore rabbit-themed NFTs on Ethereum L1 and L2. "
        "You often worry about the time. Be short, conversational, and rabbit-themed."
    ),
    model="gpt-4o-mini",
)

# ----------------------
# Session / Memory Setup
# ----------------------
DB_PATH = "/tmp/sessions.db"  # ephemeral storage on Cloud Run


class SessionManager:
    """Handles ephemeral session memory using SQLite."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages TEXT
                )
            """)

    def get_messages(self, session_id: str, limit: int = 20):
        import json
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT messages FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            if row and row["messages"]:
                messages = json.loads(row["messages"])
                return messages[-limit:]
            return []

    def save_message(self, session_id: str, role: str, text: str):
        import json
        messages = self.get_messages(session_id)
        messages.append({"role": role, "text": text})
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions(session_id, messages) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET messages=?",
                (session_id, json.dumps(messages), json.dumps(messages))
            )


session_manager = SessionManager()

# ----------------------
# Agents SDK Helper
# ----------------------
async def run_agent_with_memory(session_id: str, user_message: str):
    """Runs the OpenAI Agent with SQLite session memory."""
    # Load memory
    memory = session_manager.get_messages(session_id)

    # Convert to OpenAI message format
    contextual_messages = [{"role": m["role"], "content": m["text"]} for m in memory]

    # Add latest user message
    contextual_messages.append({"role": "user", "content": user_message})

    # Run the agent
    async with agent.runner(user_id=session_id, enable_memory=True) as runner:
        result = await runner.run(input=contextual_messages)

    return result.output


# ----------------------
# Routes
# ----------------------
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    message = data.get("message", "").strip()

    if not message:
        raise HTTPException(status_code=400, detail="message required")

    # Generate session_id if first request
    if not session_id:
        session_id = secrets.token_urlsafe(32)

    # Run agent
    reply = await run_agent_with_memory(session_id, message)

    # Save messages to SQLite
    session_manager.save_message(session_id, "user", message)
    session_manager.save_message(session_id, "assistant", reply)

    return {"response": reply, "session_id": session_id}


@app.get("/health")
def health_check():
    return {"status": "ok"}

