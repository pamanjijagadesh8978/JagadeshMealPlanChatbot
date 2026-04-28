"""
app.py  –  Streamlit frontend  (streaming edition)
===================================================
Auth redesign
─────────────
  • Credentials (username → {password, api_token}) live in a local
    credentials.json file – username/password never leave Streamlit.
  • Registration: user provides username, password, and api_token.
    The token is validated against the identity service first; if
    accepted the new entry is written to credentials.json.
  • On login Streamlit verifies the credentials locally, then sends
    only the api_token to POST /login on FastAPI.
  • FastAPI returns a short-lived JWT; every subsequent request uses
    only that JWT (Bearer header) – no username/password is ever sent.

credentials.json format
────────────────────────
  {
    "alice": {"password": "secret123", "api_token": "tok_abc…"},
    "bob":   {"password": "hunter2",   "api_token": "tok_xyz…"}
  }

Reads the NDJSON stream from FastAPI and renders:
  • Tool call status cards (start → result) in real time
  • AI reply tokens word-by-word via st.write_stream
  • Response time: measured from HTTP request fired → backend "done" event
    (excludes Streamlit token-rendering time on screen)
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import streamlit as st

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

API_URL          = "http://localhost:8000"
# API_URL = "https://jagadeshmealplanchatbot-768208692465.europe-west1.run.app"
CREDENTIALS_FILE = Path(os.getenv("CREDENTIALS_FILE", "credentials.json"))

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
)


# ─────────────────────────────────────────────
# LOCAL CREDENTIALS HELPER
# ─────────────────────────────────────────────

def _load_credentials() -> dict:
    """
    Load the credentials store from disk.
    Returns an empty dict if the file is missing or malformed.
    Format: { "<username>": { "password": "…", "api_token": "…" }, … }
    """
    if not CREDENTIALS_FILE.exists():
        return {}
    try:
        with CREDENTIALS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _verify_local_credentials(username: str, password: str) -> str | None:
    """
    Check username/password against credentials.json.
    Returns the associated api_token on success, None on failure.
    """
    store = _load_credentials()
    entry = store.get(username.strip().lower())
    if entry and entry.get("password") == password:
        return entry.get("api_token", "")
    return None


def _save_credentials(store: dict) -> bool:
    """
    Atomically write the credentials store back to credentials.json.
    Returns True on success, False on any I/O error.
    """
    try:
        tmp = CREDENTIALS_FILE.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(store, f, indent=2, ensure_ascii=False)
        tmp.replace(CREDENTIALS_FILE)   # atomic rename on most OSes
        return True
    except OSError:
        return False


def api_register(api_token: str) -> tuple[bool, str]:
    try:
        r = requests.post(
            f"{API_URL}/register",
            json={"api_token": api_token},
            timeout=20,
        )

        if r.status_code == 200:
            data = r.json()
            return data.get("success", False), data.get("message", "")
        else:
            return False, r.json().get("message", "Registration failed.")

    except requests.exceptions.ConnectionError:
        return False, "Cannot reach backend."
    
def _register_local_user(
    username: str,
    password: str,
    api_token: str,
) -> tuple[bool, str]:
    """
    Add a new entry to credentials.json.

    Returns (True, "") on success or (False, reason) on failure:
      • username already exists
      • I/O error saving the file
    """
    key   = username.strip().lower()
    store = _load_credentials()

    if key in store:
        return False, "Username already exists. Please choose a different one."

    store[key] = {"password": password, "api_token": api_token.strip()}

    if not _save_credentials(store):
        return False, "Could not write credentials file. Check file permissions."

    return True, ""


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────

def _init_session() -> None:
    defaults = {
        "token":          None,   # JWT from FastAPI
        "sessions":       [],     # [{session_id, title, created_at}]
        "active_session": None,
        "messages":       [],     # [{role, content, tools?, response_time?}]
        "last_response_time": None,  # seconds float, set by stream_chat
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _logged_in() -> bool:
    return st.session_state.token is not None


# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def _fmt_date(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
        return dt.strftime("%b %d, %H:%M")
    except Exception:
        return iso


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    secs    = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def _headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


# ─────────────────────────────────────────────
# API HELPERS  (non-streaming)
# ─────────────────────────────────────────────

def api_login(api_token: str) -> tuple[bool, str]:
    """
    Send only the api_token to FastAPI.
    FastAPI resolves the user_id from the identity service and returns a JWT.
    """
    try:
        r = requests.post(
            f"{API_URL}/login",
            json={"api_token": api_token},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            st.session_state.token = data["access_token"]
            return True, ""
        return False, r.json().get("detail", "Login failed.")
    except requests.exceptions.ConnectionError:
        return False, "Cannot reach the backend. Is main.py running?"


def api_list_sessions() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/sessions", headers=_headers(), timeout=10)
        return r.json().get("sessions", []) if r.status_code == 200 else []
    except requests.exceptions.ConnectionError:
        return []


def api_create_session(title: str = "New Chat") -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/sessions",
            json={"title": title},
            headers=_headers(),
            timeout=10,
        )
        return r.json() if r.status_code == 201 else None
    except requests.exceptions.ConnectionError:
        return None


def api_delete_session(session_id: str) -> tuple[bool, str]:
    try:
        r = requests.delete(
            f"{API_URL}/sessions/{session_id}",
            headers=_headers(),
            timeout=10,
        )
        return (True, "Session deleted.") if r.status_code == 200 \
               else (False, r.json().get("detail", "Could not delete."))
    except requests.exceptions.ConnectionError:
        return False, "Cannot reach the backend."


def api_load_history(session_id: str) -> list[dict]:
    try:
        r = requests.get(
            f"{API_URL}/history/{session_id}",
            headers=_headers(),
            timeout=10,
        )
        return r.json().get("messages", []) if r.status_code == 200 else []
    except requests.exceptions.ConnectionError:
        return []


# ─────────────────────────────────────────────
# STREAMING CONSUMER
# ─────────────────────────────────────────────

def stream_chat(session_id: str, message: str):
    """
    Open the NDJSON stream from POST /chat/{session_id} and yield
    display-ready objects that Streamlit renders live.

    Yields
    ──────
    String fragments (tokens) consumed by st.write_stream().

    Response time
    ─────────────
    The timer starts immediately before the HTTP POST is fired and stops
    the instant the backend emits the "done" event — this measures only
    the backend processing + network time, excluding Streamlit's
    token-rendering overhead.
    The result is written to st.session_state.last_response_time.
    """
    url     = f"{API_URL}/chat/{session_id}"
    headers = {**_headers(), "Accept": "application/x-ndjson"}

    tool_status_box = st.empty()
    active_tools: dict[str, dict] = {}

    def _update_tool_box():
        if not active_tools:
            tool_status_box.empty()
            return
        lines = []
        for name, info in active_tools.items():
            if info["done"]:
                lines.append(f"✅ **{name}** → `{info['output']}`")
            else:
                lines.append(f"⚙️ **{name}** — running…  \n  input: `{info['input']}`")
        tool_status_box.info("\n\n".join(lines))

    tools_used: list[str] = []
    error_msg:  str | None = None

    # ── Start timer right before the request leaves the client ──────────
    st.session_state.last_response_time = None
    t_start = time.perf_counter()

    try:
        with requests.post(
            url,
            json={"message": message},
            headers=headers,
            stream=True,
            timeout=120,
        ) as resp:

            if resp.status_code != 200:
                yield f"\n\n⚠️ Error {resp.status_code}: {resp.text}"
                return

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")

                # ── Streaming token ───────────────────────────
                if etype == "token":
                    yield event["content"]

                # ── Tool started ──────────────────────────────
                elif etype == "tool_start":
                    name = event["tool"]
                    active_tools[name] = {
                        "done":   False,
                        "input":  event.get("input", ""),
                        "output": "",
                    }
                    _update_tool_box()

                # ── Tool finished ─────────────────────────────
                elif etype == "tool_end":
                    name = event["tool"]
                    if name in active_tools:
                        active_tools[name]["done"]   = True
                        active_tools[name]["output"] = event.get("output", "")
                    _update_tool_box()

                # ── Stream complete → stop the clock here ─────
                elif etype == "done":
                    tools_used = event.get("tools_used", [])
                    # Clock stops the moment backend signals completion,
                    # before any further Streamlit rendering occurs.
                    st.session_state.last_response_time = (
                        time.perf_counter() - t_start
                    )

                # ── Error from backend ────────────────────────
                elif etype == "error":
                    error_msg = event.get("detail", "Unknown error")
                    yield f"\n\n⚠️ {error_msg}"

    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ Cannot reach the backend. Is main.py running?"
        return

    # Collapse tool box to a compact summary after streaming ends
    if tools_used:
        tool_status_box.caption(
            "Tools used: " + " · ".join(f"`{t}`" for t in tools_used)
        )
    else:
        tool_status_box.empty()


# ─────────────────────────────────────────────
# STATE MUTATORS
# ─────────────────────────────────────────────

def _switch_session(session_id: str) -> None:
    st.session_state.active_session = session_id
    history = api_load_history(session_id)
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]} for m in history
    ]


def _start_new_chat() -> None:
    session = api_create_session("New Chat")
    if session:
        st.session_state.sessions.insert(0, session)
        st.session_state.active_session = session["session_id"]
        st.session_state.messages = []


def _logout() -> None:
    for key in ["token", "sessions", "active_session", "messages"]:
        st.session_state[key] = [] if key in ("sessions", "messages") else None


# ─────────────────────────────────────────────
# PAGE: AUTH
# ─────────────────────────────────────────────

def render_auth_page() -> None:
    st.title("🤖 AI Chatbot")
    st.write("Welcome! Please log in or create an account to start chatting.")
    st.divider()

    tab_login, tab_register = st.tabs(["Login", "Register"])

    # ── LOGIN TAB ─────────────────────────────────
    with tab_login:
        st.subheader("Sign in to your account")
        username = st.text_input("Username", key="login_user",
                                 placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_pass",
                                 placeholder="Enter your password")

        if st.button("Login", use_container_width=True, key="btn_login"):
            if not username or not password:
                st.warning("Please fill in both fields.")
            else:
                # Step 1 – verify locally against credentials.json
                api_token = _verify_local_credentials(username, password)
                if api_token is None:
                    st.error("Invalid username or password.")
                else:
                    # Step 2 – exchange api_token for a FastAPI JWT
                    with st.spinner("Authenticating…"):
                        ok, msg = api_login(api_token)
                    if ok:
                        # Always load existing sessions into sidebar,
                        # then immediately open a fresh new chat.
                        st.session_state.sessions = api_list_sessions()
                        _start_new_chat()
                        st.rerun()
                    else:
                        st.error(msg)

        st.caption(
            "ℹ️ Credentials are verified locally. "
            "Your username and password are never sent to the server."
        )

    # ── REGISTER TAB ──────────────────────────────
    with tab_register:
        st.subheader("Create a new account")
        new_user  = st.text_input("Choose a username", key="reg_user",
                                  placeholder="At least 3 characters")
        new_pass  = st.text_input("Choose a password", type="password",
                                  key="reg_pass",
                                  placeholder="At least 6 characters")
        conf_pass = st.text_input("Confirm password", type="password",
                                  key="reg_conf",
                                  placeholder="Repeat your password")
        api_token = st.text_input("API Token", type="password",
                                  key="reg_token",
                                  placeholder="Your API server token",
                                  help="The token used to identify you on the backend.")

        if st.button("Create Account", use_container_width=True, key="btn_register"):
            # ── Validation ────────────────────────
            if not new_user or not new_pass or not conf_pass or not api_token:
                st.warning("Please fill in all fields.")
            elif len(new_user.strip()) < 3:
                st.error("Username must be at least 3 characters.")
            elif len(new_pass) < 6:
                st.error("Password must be at least 6 characters.")
            elif new_pass != conf_pass:
                st.error("Passwords do not match.")
            else:
                # ── Verify token with FastAPI / identity service ──
                with st.spinner("Verifying token with identity service…"):
                    with st.spinner("Creating your profile…"):
                        ok, msg = api_register(api_token.strip())

                if not ok:
                    st.error(f"Token rejected by the server: {msg}")
                else:
                    # Token is valid – save to credentials.json
                    # Log out of the JWT we just acquired; user must log in properly
                    st.session_state.token = None

                    saved_ok, save_msg = _register_local_user(
                        new_user, new_pass, api_token
                    )
                    if saved_ok:
                        st.success(
                            f"Account **{new_user.strip().lower()}** created! "
                            "You can now log in."
                        )
                    else:
                        st.error(save_msg)

        st.caption(
            "ℹ️ Your username and password are stored only in the local "
            "credentials.json file. Only your token is sent to the backend."
        )


# ─────────────────────────────────────────────
# PAGE: CHAT
# ─────────────────────────────────────────────

def render_chat_page() -> None:
    active_id = st.session_state.active_session
    sessions  = st.session_state.sessions

    # ── Sidebar ───────────────────────────────────
    with st.sidebar:
        st.divider()

        if st.button("＋  New Chat", use_container_width=True, key="btn_new_chat"):
            _start_new_chat()
            st.rerun()

        st.divider()

        if sessions:
            st.caption("Your conversations")
            for s in sessions:
                sid       = s["session_id"]
                is_active = sid == active_id
                col_btn, col_del = st.columns([5, 1])

                with col_btn:
                    label = f"{'▶ ' if is_active else ''}{s['title']}\n{_fmt_date(s['created_at'])}"
                    if st.button(
                        label,
                        key=f"sess_{sid}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        if not is_active:
                            _switch_session(sid)
                            st.rerun()

                with col_del:
                    if st.button("🗑", key=f"del_{sid}", help="Delete this chat"):
                        ok, _ = api_delete_session(sid)
                        if ok:
                            st.session_state.sessions = [
                                x for x in st.session_state.sessions
                                if x["session_id"] != sid
                            ]
                            if sid == active_id:
                                remaining = st.session_state.sessions
                                if remaining:
                                    _switch_session(remaining[0]["session_id"])
                                else:
                                    _start_new_chat()
                            st.rerun()
        else:
            st.caption("No conversations yet.")

        st.divider()
        if st.button("Logout", use_container_width=True, key="btn_logout"):
            _logout()
            st.rerun()
        st.caption("Tools: calculator · get_weather · search_web")

    # ── Main header ───────────────────────────────
    st.title("🤖 AI Chatbot")

    # ── Auto-create a new chat if none is active ──
    # Handles page refresh: active_session is wiped but JWT persists,
    # so we silently open a fresh chat instead of showing a dead end.
    if not active_id:
        _start_new_chat()
        st.rerun()

    active_meta = next((s for s in sessions if s["session_id"] == active_id), None)
    if active_meta:
        st.caption(
            f"Chat: {active_meta['title']}  ·  {_fmt_date(active_meta['created_at'])}"
        )

    st.divider()

    # ── Render existing message history ───────────
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "human" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])
            # Show tools and response time on AI messages
            if msg["role"] == "ai":
                meta_parts = []
                if msg.get("tools"):
                    meta_parts.append(
                        "Tools: " + " · ".join(f"`{t}`" for t in msg["tools"])
                    )
                if msg.get("response_time") is not None:
                    meta_parts.append(
                        f"⏱ Response time: {_fmt_elapsed(msg['response_time'])}"
                    )
                if meta_parts:
                    st.caption("  |  ".join(meta_parts))

    # ── Chat input ────────────────────────────────
    user_input = st.chat_input("Type your message…")

    if user_input:
        # 1. Show user bubble immediately
        st.session_state.messages.append({"role": "human", "content": user_input})
        with st.chat_message("human", avatar="🧑"):
            st.write(user_input)

        # 2. Open AI bubble and stream into it.
        #    The timer runs inside stream_chat — starts on HTTP POST,
        #    stops on backend "done" event, stored in session_state.
        with st.chat_message("assistant", avatar="🤖"):
            full_reply = st.write_stream(stream_chat(active_id, user_input))

            # Read the precise backend response time recorded by stream_chat
            t_elapsed = st.session_state.last_response_time
            if t_elapsed is not None:
                st.caption(f"⏱ Response time: {_fmt_elapsed(t_elapsed)}")

        # 3. Persist assembled reply + response time in session state
        st.session_state.messages.append({
            "role":          "ai",
            "content":       full_reply or "",
            "response_time": t_elapsed,
        })

        # 4. Auto-rename "New Chat" sessions to the first message snippet
        if active_meta and active_meta["title"] == "New Chat":
            snippet = user_input[:30] + ("…" if len(user_input) > 30 else "")
            for s in st.session_state.sessions:
                if s["session_id"] == active_id:
                    s["title"] = snippet
                    break


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    _init_session()
    if _logged_in():
        render_chat_page()
    else:
        render_auth_page()


if __name__ == "__main__":
    main()
