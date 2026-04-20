import json
import logging
import os
import secrets
import aiosqlite
import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from graph import TOOLS, build_graph, build_llm
from utils import get_user_id_sql_server  # must exist in utils.py

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

_raw_secret = os.getenv("JWT_SECRET_KEY", "")
if not _raw_secret:
    _raw_secret = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET_KEY not set – using a random per-process secret.  "
        "All existing JWTs will be invalidated on restart.  "
        "Set JWT_SECRET_KEY in production."
    )

SECRET_KEY      = _raw_secret
ALGORITHM       = "HS256"
TOKEN_TTL_HOURS = 8
USERS_DB        = "users.db"
CHAT_DB         = "chat_history.db"
MEMORY_DB       = "memory.db"
PROFILE_DB      = "profiles.db"

# ─────────────────────────────────────────────
# SHARED LOCKS (imported by graph.py)
# ─────────────────────────────────────────────

users_db_lock  = asyncio.Lock()
memory_db_lock = asyncio.Lock()
profile_db_lock = asyncio.Lock()

# ─────────────────────────────────────────────
# GRAPH STATE (module-level dict, populated in lifespan)
# ─────────────────────────────────────────────
# Keys:
#   graph        – compiled LangGraph
#   checkpointer – AsyncSqliteSaver instance (entered context manager)
#   _cm_ctx      – the context manager itself (for __aexit__ on shutdown)
#   plain_llm    – raw ChatMistralAI (no tools bound; used by memory node)
#   users_conn   – shared aiosqlite connection for users/sessions DB
#   memory_conn  – shared aiosqlite connection for memory DB

graph_state: dict = {}


# ─────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    plain_llm      = build_llm()
    llm_with_tools = plain_llm.bind_tools(TOOLS)

    # ── LangGraph checkpointer ───────────────────────────────────────────
    cm  = AsyncSqliteSaver.from_conn_string(CHAT_DB)
    checkpointer = await cm.__aenter__()

    # ── Shared DB connections ────────────────────────────────────────────
    users_conn  = await aiosqlite.connect(USERS_DB)
    memory_conn = await aiosqlite.connect(MEMORY_DB)
    profile_conn = await aiosqlite.connect(PROFILE_DB)

    for conn in (users_conn, memory_conn):
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA synchronous=NORMAL;")
        await profile_conn.execute("PRAGMA foreign_keys = ON;")

    # ── Populate graph_state BEFORE calling helpers that read from it ────
    graph_state.update({
        "graph":        build_graph(llm_with_tools, checkpointer),
        "checkpointer": checkpointer,
        "_cm_ctx":      cm,
        "plain_llm":    plain_llm,
        "users_conn":   users_conn,
        "memory_conn":  memory_conn,
        "profile_conn": profile_conn,
    })

    await _init_users_db()

    logger.info("✅ LangGraph chatbot API ready.")
    yield

    # ── Teardown ─────────────────────────────────────────────────────────
    await users_conn.close()
    await memory_conn.close()
    await cm.__aexit__(None, None, None)


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

_allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()]
if not _allowed_origins:
    # Development default – disable credentials in this mode
    _allowed_origins = ["*"]
    _allow_credentials = False
    logger.warning(
        "ALLOWED_ORIGINS not set – CORS is open to all origins and "
        "credentials are disabled.  Set ALLOWED_ORIGINS in production."
    )
else:
    _allow_credentials = True

app = FastAPI(title="LangGraph Chatbot API (streaming)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


# ─────────────────────────────────────────────
# HELPERS – database
# ─────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _init_users_db() -> None:
    """Create tables if they don't exist yet."""
    users_conn  = graph_state["users_conn"]
    memory_conn = graph_state["memory_conn"]
    profile_conn = graph_state["profile_conn"]

    async with users_db_lock:
        await users_conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                api_token  TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        await users_conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                api_token  TEXT NOT NULL,
                title      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (api_token) REFERENCES users(api_token)
            )
        """)
        await users_conn.commit()

    async with memory_db_lock:
        await memory_conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT    NOT NULL,
                is_mealplan INTEGER   NOT NULL,
                content    TEXT    NOT NULL,
                embedding  BLOB    NOT NULL,
                created_at TEXT    NOT NULL
            )
        """)
        await memory_conn.commit()

    async with profile_db_lock:
        await profile_conn.execute("""
            CREATE TABLE IF NOT EXISTS UserHealthProfile (
                user_id TEXT PRIMARY KEY,

                Name TEXT NOT NULL,
                Age INTEGER,
                Gender TEXT,

                WeightKG REAL,
                HeightCM REAL,
                WaistCircumferenceCM REAL,

                Cuisine TEXT,
                ActivityLevel TEXT,
                Calories INTEGER,

                DietaryPreference TEXT,
                Restrictions TEXT,
                DigestiveIssues TEXT,
                Allergies TEXT,
                SymptomAggravatingFoods TEXT,

                HeartRate TEXT,
                BloodPressure TEXT,
                BodyTemperature TEXT,
                BloodOxygen TEXT,
                RespiratoryRate TEXT,

                MedicalConditions TEXT,
                Goals TEXT,

                CreatedDate TEXT DEFAULT (datetime('now')),
                ModifiedDate TEXT
            )
        """)

    await profile_conn.commit()


async def _get_user_by_token(api_token: str) -> dict | None:
    conn   = graph_state["users_conn"]
    cursor = await conn.execute(
        "SELECT api_token, user_id FROM users WHERE api_token = ?",
        (api_token,),
    )
    row = await cursor.fetchone()
    return {"api_token": row[0], "user_id": row[1]} if row else None


async def _upsert_user(api_token: str, user_id: str) -> None:
    conn = graph_state["users_conn"]
    async with users_db_lock:
        await conn.execute(
            """
            INSERT INTO users (api_token, user_id, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(api_token) DO UPDATE SET user_id = excluded.user_id
            """,
            (api_token, user_id, _now()),
        )
        await conn.commit()


async def _create_session_record(api_token: str, title: str) -> str:
    session_id = str(uuid.uuid4())
    conn       = graph_state["users_conn"]
    async with users_db_lock:
        await conn.execute(
            "INSERT INTO sessions (session_id, api_token, title, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, api_token, title, _now()),
        )
        await conn.commit()
    return session_id


async def _list_sessions(api_token: str) -> list[dict]:
    conn   = graph_state["users_conn"]
    cursor = await conn.execute(
        "SELECT session_id, title, created_at FROM sessions "
        "WHERE api_token = ? ORDER BY created_at DESC",
        (api_token,),
    )
    rows = await cursor.fetchall()
    return [{"session_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


async def _get_session(session_id: str) -> dict | None:
    conn   = graph_state["users_conn"]
    cursor = await conn.execute(
        "SELECT session_id, api_token, title, created_at "
        "FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    return {"session_id": row[0], "api_token": row[1], "title": row[2], "created_at": row[3]}


async def _delete_session_record(session_id: str) -> None:
    conn = graph_state["users_conn"]
    async with users_db_lock:
        await conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await conn.commit()


def _thread_id(api_token: str, session_id: str) -> str:
    """Namespaced LangGraph thread – guarantees cross-user isolation."""
    return f"{api_token}::{session_id}"


async def _assert_owns_session(session_id: str, api_token: str) -> dict:
    """Return session dict; raise 404 for missing OR foreign sessions."""
    session = await _get_session(session_id)
    if not session or session["api_token"] != api_token:
        raise HTTPException(404, "Session not found.")
    return session


# ─────────────────────────────────────────────
# HELPERS – JWT
# ─────────────────────────────────────────────

def _create_jwt(api_token: str) -> str:
    expire  = datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL_HOURS)
    payload = {"sub": api_token, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def _decode_jwt(token: str) -> str:
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return data["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token.")


def get_current_api_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    return _decode_jwt(credentials.credentials)


# ─────────────────────────────────────────────
# HELPERS – NDJSON
# ─────────────────────────────────────────────

def _ndjson(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


# ─────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────

class LoginRequest(BaseModel):
    api_token: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"

class NewSessionRequest(BaseModel):
    title: str = "New Chat"

class SessionItem(BaseModel):
    session_id: str
    title:      str
    created_at: str

class SessionsResponse(BaseModel):
    sessions: list[SessionItem]

class NewSessionResponse(BaseModel):
    session_id: str
    title:      str
    created_at: str

class ChatRequest(BaseModel):
    message: str

class MessageItem(BaseModel):
    role:    str
    content: str

class HistoryResponse(BaseModel):
    session_id: str
    title:      str
    messages:   list[MessageItem]


# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────

@app.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    """
    Streamlit calls this after verifying credentials locally.
    We resolve user_id via the external identity service, upsert the
    record in users.db, then issue a short-lived JWT.
    """
    api_token = body.api_token.strip()
    if not api_token:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "api_token is required.")

    try:
        user_id = await get_user_id_sql_server(api_token)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Could not verify token with identity service: {exc}")

    if not user_id:
        raise HTTPException(401, "api_token rejected by identity service.")

    await _upsert_user(api_token, user_id)
    return TokenResponse(access_token=_create_jwt(api_token))


# ─────────────────────────────────────────────
# SESSION ROUTES
# ─────────────────────────────────────────────

@app.post("/sessions", response_model=NewSessionResponse, status_code=201)
async def create_session(
    body:      NewSessionRequest,
    api_token: str = Depends(get_current_api_token),
):
    title      = body.title.strip() or "New Chat"
    session_id = await _create_session_record(api_token, title)
    session    = await _get_session(session_id)
    return NewSessionResponse(
        session_id=session["session_id"],
        title=session["title"],
        created_at=session["created_at"],
    )


@app.get("/sessions", response_model=SessionsResponse)
async def list_sessions(api_token: str = Depends(get_current_api_token)):
    sessions = await _list_sessions(api_token)
    return SessionsResponse(sessions=[SessionItem(**s) for s in sessions])

@app.delete("/sessions/{session_id}", status_code=200)
async def delete_session(
    session_id: str,
    api_token:  str = Depends(get_current_api_token),
):
    await _assert_owns_session(session_id, api_token)

    checkpointer = graph_state["checkpointer"]
    thread       = _thread_id(api_token, session_id)

    try:
        async with checkpointer.conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (thread,)
        ):
            pass
        async with checkpointer.conn.execute(
            "DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread,)
        ):
            pass
        await checkpointer.conn.commit()
    except Exception as exc:
        logger.warning("Could not delete checkpoints for thread %s: %s", thread, exc)

    await _delete_session_record(session_id)
    return {"message": "Session deleted."}

# ─────────────────────────────────────────────
# STREAMING CHAT ROUTE
# ─────────────────────────────────────────────

@app.post("/chat/{session_id}")
async def chat(
    session_id: str,
    body:       ChatRequest,
    api_token:  str = Depends(get_current_api_token),
):
    """
    Stream LangGraph events back to the client as NDJSON lines.

    Event shape (one JSON object per line):
      {"type": "tool_start",  "tool": "calculator", "input": "..."}
      {"type": "tool_end",    "tool": "calculator",  "output": "42"}
      {"type": "token",       "content": "The answer"}
      {"type": "done",        "tools_used": ["calculator"]}
      {"type": "error",       "detail": "..."}
    """
    await _assert_owns_session(session_id, api_token)

    user_record = await _get_user_by_token(api_token)
    if not user_record:
        raise HTTPException(401, "Unknown api_token – please log in again.")
    user_id = user_record["user_id"]

    graph  = graph_state["graph"]
    thread = _thread_id(api_token, session_id)
    config = {"configurable": {"thread_id": thread}}

    async def event_generator() -> AsyncGenerator[str, None]:
        tools_used: list[str] = []

        try:
            async for event in graph.astream_events(
                {
                    "messages": [HumanMessage(content=body.message)],
                    "user_id":  user_id,
                },
                config=config,
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    if node_name != "chatbot":
                        continue

                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        raw   = chunk.content
                        token = raw if isinstance(raw, str) else (raw[0].get("text", "") if raw else "")
                        if token:
                            yield _ndjson({"type": "token", "content": token})

                elif kind == "on_tool_start":
                    tool_name  = event.get("name", "unknown_tool")
                    tool_input = event.get("data", {}).get("input", {})
                    if tool_name not in tools_used:
                        tools_used.append(tool_name)
                    yield _ndjson({"type": "tool_start", "tool": tool_name, "input": str(tool_input)})

                elif kind == "on_tool_end":
                    tool_name   = event.get("name", "unknown_tool")
                    tool_output = event.get("data", {}).get("output", "")
                    yield _ndjson({"type": "tool_end", "tool": tool_name, "output": str(tool_output)})

            yield _ndjson({"type": "done", "tools_used": tools_used})

        except Exception as exc:
            logger.exception("Error during streaming chat for session %s", session_id)
            yield _ndjson({"type": "error", "detail": str(exc)})
            yield _ndjson({"type": "done", "tools_used": tools_used})

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────
# HISTORY ROUTE
# ─────────────────────────────────────────────

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(
    session_id: str,
    api_token:  str = Depends(get_current_api_token),
):
    session = await _assert_owns_session(session_id, api_token)

    graph  = graph_state["graph"]
    thread = _thread_id(api_token, session_id)
    config = {"configurable": {"thread_id": thread}}

    try:
        state        = await graph.aget_state(config)
        raw_messages = state.values.get("messages", []) if state.values else []
    except Exception:
        raw_messages = []

    messages: list[MessageItem] = []
    for msg in raw_messages:
        role    = "ai" if isinstance(msg, AIMessage) else "human"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if content:
            messages.append(MessageItem(role=role, content=content))

    return HistoryResponse(
        session_id=session_id,
        title=session["title"],
        messages=messages,
    )


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": _now()}


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
