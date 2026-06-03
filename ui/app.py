# ui/app.py
"""
Interfaz Streamlit para el Vision MCP Agent.
Páginas:
  🔍 Buscar Imágenes  → búsqueda por descripción o imagen
  🗂️ Repositorio      → gestión del repositorio
  💬 Chat con Agente  → conversación multi-turno con el agente
  ℹ️ Estado           → health check del sistema
"""
import streamlit as st

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision MCP Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .stApp { background-color: #0d0f14; color: #e8e8e8; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #141720 100%);
        border: 1px solid #2d3050;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
    }
    .result-card {
        background: #141720;
        border: 1px solid #2d3050;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.75rem 0;
        transition: border-color 0.2s;
    }
    .result-card:hover { border-color: #5b7fff; }

    .score-badge {
        display: inline-block;
        background: linear-gradient(90deg, #5b7fff, #8b5cf6);
        color: white;
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        margin-bottom: 0.5rem;
    }
    .tag-badge {
        display: inline-block;
        background: #1e2235;
        color: #8892b0;
        font-size: 0.72rem;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        margin: 0.1rem;
        border: 1px solid #2d3050;
    }
    .tool-pill {
        display: inline-block;
        background: #0d2137;
        color: #4fc3f7;
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        border: 1px solid #1a4a6e;
        margin: 0.15rem;
    }
    .agent-response {
        background: #121a2e;
        border-left: 3px solid #5b7fff;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .status-ok { color: #4ade80; }
    .status-error { color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ── Imports de páginas ────────────────────────────────────────────────────────
from ui.views.search import render_search_page
from ui.views.repository import render_repository_page
from ui.views.chat import render_chat_page
from ui.views.status import render_status_page

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Vision MCP Agent")
    st.markdown("*Búsqueda visual por memoria*")
    st.divider()

    page = st.radio(
        "Navegación",
        ["🔍 Buscar Imágenes", "🗂️ Repositorio", "💬 Chat con Agente", "ℹ️ Estado"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("""
    **Stack:**
    - 🧠 Ollama (LLM)
    - 🖼️ CLIP (embeddings)
    - 🤖 BLIP VQA
    - 🔧 MCP (herramientas)
    """)

    # Config rápida
    with st.expander("⚙️ Configuración"):
        api_url = st.text_input(
            "API URL",
            value=st.session_state.get("api_url", "http://api:8001"),
        )
        st.session_state["api_url"] = api_url
        top_k = st.slider("Resultados máximos", 1, 20, 5)
        st.session_state["top_k"] = top_k
        threshold = st.slider("Umbral de similitud", 0.0, 1.0, 0.20, 0.05)
        st.session_state["threshold"] = threshold

# ── Router de páginas ─────────────────────────────────────────────────────────
if page == "🔍 Buscar Imágenes":
    render_search_page()
elif page == "🗂️ Repositorio":
    render_repository_page()
elif page == "💬 Chat con Agente":
    render_chat_page()
else:
    render_status_page()
