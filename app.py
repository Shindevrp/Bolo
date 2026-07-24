from __future__ import annotations

import json
import urllib.request

import streamlit as st

st.set_page_config(page_title="TASA", layout="centered")

st.markdown(
    """
<style>
    .stApp { background: radial-gradient(ellipse at 50% 0%, #0B1A2E 0%, #020617 70%); }
    .stApp > header { display: none; }
    #MainMenu, footer { display: none; }
    .block-container { padding: 1rem 1.5rem !important; max-width: 680px !important; }

    .card {
        background: rgba(15, 25, 45, 0.45);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 24px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }

    div[data-testid="stChatMessageContent"] p { margin: 0; }
    .stChatMessage { background: transparent !important; padding: 0.25rem 0 !important; }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) { justify-content: flex-end; }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"] {
        background: rgba(59,130,246,0.12) !important;
        border: 1px solid rgba(59,130,246,0.15);
        border-radius: 18px 18px 4px 18px !important;
        padding: 0.4rem 0.9rem !important;
        color: rgba(255,255,255,0.85) !important;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) div[data-testid="stChatMessageContent"] {
        background: rgba(16,185,129,0.08) !important;
        border: 1px solid rgba(16,185,129,0.12);
        border-radius: 18px 18px 18px 4px !important;
        padding: 0.4rem 0.9rem !important;
        color: rgba(255,255,255,0.8) !important;
    }

    .stChatInputContainer {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,0.03) !important;
    }
    .stChatInputContainer textarea { color: rgba(255,255,255,0.85) !important; font-size: 0.9rem !important; }
    .stChatInputContainer textarea::placeholder { color: rgba(255,255,255,0.2) !important; }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;margin:.5rem 0'>"
    '<a href="http://localhost:8000" target="_blank" '
    'style="display:inline-flex;align-items:center;gap:8px;'
    'background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.2);'
    'border-radius:14px;padding:.6rem 1.2rem;color:rgba(255,255,255,.8);'
    'text-decoration:none;font-size:.85rem;transition:all .2s">'
    "🎤 Open Voice UI in new tab"
    "</a></p>",
    unsafe_allow_html=True,
)

prompt = st.chat_input("Type a message...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("<span style='color:rgba(255,255,255,0.3);font-size:0.85rem'>...</span>", unsafe_allow_html=True)
        data = json.dumps({"message": prompt}).encode()
        req = urllib.request.Request(
            "http://localhost:8000/chat/stream",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        full = ""
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line_bytes in resp:
                line = line_bytes.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                if payload["type"] == "token":
                    full += payload["token"]
                    placeholder.markdown(full + "▌")
                elif payload["type"] == "done":
                    break
        placeholder.markdown(full)
    st.session_state.messages.append({"role": "assistant", "content": full})
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
