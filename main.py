from fastapi import FastAPI, Request, HTTPException
import os
import sqlite3
import secrets
import logging
from openai import OpenAI
from agents import Agent

# ----------------------
# Logging Setup
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------
# Configuration
# ----------------------
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
        logger.info("Database initialized successfully")

    def get_messages(self, session_id: str, limit: int = 20):
        import json
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT messages FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            if row and row["messages"]:
                messages = json.loads(row["messages"])
                logger.info(f"Loaded {len(messages)} messages for session {session_id}")
                return messages[-limit:]
            logger.info(f"No existing messages for session {session_id}")
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
        logger.info(f"Saved {role} message for session {session_id}")


session_manager = SessionManager()

# ----------------------
# Agents SDK Helper
# ----------------------
async def run_agent_with_memory(session_id: str, user_message: str):
    """Runs the OpenAI Agent with SQLite session memory."""
    logger.info(f"Running agent for session {session_id}")
    logger.info(f"User message: {user_message}")
    
    # Load memory
    memory = session_manager.get_messages(session_id)

    # Convert to OpenAI message format
    contextual_messages = [{"role": m["role"], "content": m["text"]} for m in memory]

    # Add latest user message
    contextual_messages.append({"role": "user", "content": user_message})
    
    logger.info(f"Total messages in context: {len(contextual_messages)}")

    try:
        # Run the agent
        async with agent.runner(user_id=session_id, enable_memory=True) as runner:
            result = await runner.run(input=contextual_messages)
        
        # Log the raw result for debugging
        logger.info(f"Agent result type: {type(result)}")
        logger.info(f"Agent result attributes: {dir(result)}")
        logger.info(f"Agent result: {result}")
        
        # Extract the text content from the result
        response_text = None
        
        # Try to extract from messages
        if hasattr(result, 'messages') and result.messages:
            logger.info(f"Found {len(result.messages)} messages in result")
            for msg in reversed(result.messages):
                logger.info(f"Message role: {msg.role}, content type: {type(msg.content)}")
                if msg.role == "assistant" and msg.content:
                    if isinstance(msg.content, list):
                        response_text = msg.content[0].text if hasattr(msg.content[0], 'text') else str(msg.content[0])
                    else:
                        response_text = msg.content
                    break
        
        # Try to extract from output
        if not response_text and hasattr(result, 'output'):
            logger.info(f"Trying output attribute: {result.output}")
            response_text = str(result.output) if result.output else None
        
        # Final fallback
        if not response_text:
            logger.warning("Could not extract response from agent result")
            response_text = "I'm having trouble responding right now. Please try again!"
        
        logger.info(f"Final response: {response_text}")
        return response_text
        
    except Exception as e:
        logger.exception(f"Error running agent: {e}")
        return "Sorry, I encountered an error. Please try again!"


# ----------------------
# Routes
# ----------------------
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        message = data.get("message", "").strip()

        logger.info(f"Received chat request - session_id: {session_id}, message: {message}")

        if not message:
            raise HTTPException(status_code=400, detail="message required")

        # Generate session_id if first request
        if not session_id:
            session_id = secrets.token_urlsafe(32)
            logger.info(f"Generated new session_id: {session_id}")

        # Run agent
        reply = await run_agent_with_memory(session_id, message)

        # Save messages to SQLite
        session_manager.save_message(session_id, "user", message)
        session_manager.save_message(session_id, "assistant", reply)

        logger.info(f"Sending response: {reply}")
        return {"response": reply, "session_id": session_id}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
def health_check():
    return {"status": "ok"}
