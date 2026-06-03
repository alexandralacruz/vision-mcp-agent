# streamlit_app/pages/status.py
"""Página de estado del sistema."""
import requests
import streamlit as st


def _get_api_url() -> str:
    return st.session_state.get("api_url", "http://api:8001")


def render_status_page():
    st.title("ℹ️ Estado del Sistema")
    api_url = _get_api_url()
    
    # st.caption(f"Conectado a API en `{api_url}`")


    if st.button("🔄 Verificar estado"):
        st.rerun()

    try:
        resp = requests.get(f"{api_url}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()

            col1, col2, col3 = st.columns(3)
            with col1:
                status = "🟢 Online" if data.get("ollama_available") else "🔴 Offline"
                st.metric("Ollama", status)
                st.caption(f"Modelo: {data.get('ollama_model', 'N/A')}")
            with col2:
                st.metric("Imágenes en repo", data.get("repository_images", 0))
                st.caption("Imágenes indexadas")
            with col3:
                st.metric("API", "🟢 Online")
                st.caption(f"URL: {api_url}")

            st.divider()
            st.markdown("### Modelos ML")
            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"**CLIP:** `{data.get('clip_model', 'N/A')}`\n\nEmbeddings visuales y de texto en espacio compartido.")
            with col_b:
                st.info(f"**BLIP VQA:** `{data.get('blip_model', 'N/A')}`\n\nResponde preguntas sobre el contenido visual.")

            if not data.get("ollama_available"):
                st.warning("""
                **Ollama no está disponible.** Para iniciarlo:
                ```bash
                ollama serve
                ollama pull llama3.2-vision
                ```
                """)
        else:
            st.error(f"API respondió con error {resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error(f"❌ No se puede conectar a la API en `{api_url}`")
        st.markdown("""
        **Para iniciar la API:**
        ```bash
        uvicorn api.main:app --reload --port 8001
        ```
        """)
    except Exception as e:
        st.error(f"Error: {e}")

    st.divider()
    st.markdown("""
    ### Arquitectura MCP

    ```
    Usuario (Streamlit / FastAPI)
           ↓
    Agente Ollama (llama3.2-vision)
           ↓ invoca herramientas MCP
    ┌──────┴──────────────────┐
    │                         │
    clip_search            vqa_query
    (CLIP embeddings)    (BLIP VQA)
    │                         │
    └──────┬──────────────────┘
           ↓
    Repositorio de imágenes
    (embeddings + metadatos JSON)
    ```

    ### Flujo de indexado
    1. Usuario sube imagen → API
    2. CLIP genera embedding vectorial (512 dims)
    3. BLIP VQA responde 10 preguntas estándar
    4. Metadatos + embedding guardados en disco

    ### Flujo de búsqueda
    1. Usuario escribe descripción
    2. Agente Ollama decide usar `clip_search`
    3. CLIP convierte texto a vector y busca similares
    4. Agente puede usar `vqa_query` para verificar
    5. Respuesta sintetizada al usuario
    """)
