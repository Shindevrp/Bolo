from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="TASA", layout="centered")

st.markdown(
    """
<style>
    .stApp { background: radial-gradient(ellipse at 50% 0%, #0B1A2E 0%, #020617 70%); }
    .stApp > header { display: none; }
    #MainMenu, footer { display: none; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    iframe { border: none; width: 100%; height: 100vh; }
</style>
""",
    unsafe_allow_html=True,
)

st.iframe("http://localhost:8000/ui", height=800)
