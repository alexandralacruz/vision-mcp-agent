# streamlit_app/pages/chat.py
"""Página de chat multi-turno con el agente."""
import requests
import streamlit as st


def _get_api_url() -> str:
    return st.session_state.get("api_url", "http://api:8001")


def render_chat_page():
    st.title("💬 Chat con el Agente")
    st.markdown("*Conversación multi-turno con el agente de visión Ollama*")

    api_url = _get_api_url()
    st.caption(f"Conectado a API en `{api_url}`")

    # Inicializar historial
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []

    # Botón limpiar
    if st.button("🗑️ Nueva conversación"):
        st.session_state.chat_history = []
        st.session_state.agent_history = []
        st.rerun()

    # Mostrar historial de chat
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        tool_calls = msg.get("tool_calls", [])
        results = msg.get("search_results", [])

        with st.chat_message(role):
            st.markdown(content)

            if tool_calls:
                tools_str = " → ".join(f"🔧 `{tc['tool']}`" for tc in tool_calls)
                st.caption(f"Herramientas usadas: {tools_str}")

            if results:
                with st.expander(f"📊 {len(results)} imágenes encontradas"):
                    for r in results[:3]:
                        st.markdown(
                            f"- **`{r.get('image_id','')[:8]}...`** "
                            f"| Score: {r.get('score', 0):.2%} "
                            f"| {r.get('filename', '')}"
                        )

    # Input del usuario
    user_input = st.chat_input("Describe la imagen que buscas...")

    if user_input:
        # Agregar al historial visual
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Agregar al historial del agente (formato Ollama)
        st.session_state.agent_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("El agente está buscando..."):
                try:
                    resp = requests.post(
                        f"{api_url}/search",
                        json={
                            "query": user_input,
                            "top_k": st.session_state.get("top_k", 5),
                            "threshold": st.session_state.get("threshold", 0.20),
                            "use_agent": True,
                            "conversation_history": st.session_state.agent_history[:-1],
                        },
                        timeout=120,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        response_text = data.get("response", "No se obtuvo respuesta.")
                        tool_calls = data.get("tool_calls", [])
                        search_results = data.get("search_results", [])

                        st.markdown(response_text)

                        if tool_calls:
                            tools_str = " → ".join(f"🔧 `{tc['tool']}`" for tc in tool_calls)
                            st.caption(f"Herramientas: {tools_str} | Iteraciones: {data.get('iterations', 1)}")

                        if search_results:
                            with st.expander(f"📊 {len(search_results)} imágenes encontradas"):
                                for r in search_results[:5]:
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        st.markdown(f"Score: **{r.get('score', 0):.2%}**")
                                    with col2:
                                        st.markdown(
                                            f"`{r.get('image_id','')[:8]}...` — {r.get('filename', '')}"
                                        )
                                        desc = r.get("description", r.get("searchable_text", ""))
                                        if desc:
                                            st.caption(desc[:150])

                        # Guardar en historial
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response_text,
                            "tool_calls": tool_calls,
                            "search_results": search_results,
                        })
                        st.session_state.agent_history.append({
                            "role": "assistant",
                            "content": response_text,
                        })

                    elif resp.status_code == 503:
                        st.error("⚠️ Ollama no disponible.")
                    else:
                        st.error(f"Error {resp.status_code}")

                except requests.exceptions.ConnectionError:
                    st.error(f"❌ No se puede conectar a la API en {api_url}")
                except Exception as e:
                    st.error(f"Error: {e}")
