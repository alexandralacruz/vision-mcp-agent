# streamlit_app/pages/search.py
"""Página de búsqueda de imágenes."""
import base64
import io
import requests
import streamlit as st
from PIL import Image


def _get_api_url() -> str:
    return st.session_state.get("api_url", "http://api:8001")


def _render_result_card(result: dict, api_url: str, idx: int):
    """Renderiza una tarjeta de resultado de búsqueda."""
    score = result.get("score", 0)
    image_id = result.get("image_id", "")
    filename = result.get("filename", "")
    description = result.get("description", result.get("searchable_text", ""))
    tags = result.get("tags", [])

    with st.container():
        col_img, col_info = st.columns([1, 2])

        with col_img:
            # Cargar imagen desde API
            try:
                img_resp = requests.get(f"{api_url}/images/{image_id}", timeout=10)
                if img_resp.status_code == 200:
                    img = Image.open(io.BytesIO(img_resp.content))
                    # st.image(img, use_column_width=True, caption=f"#{idx + 1}") --deprecated
                    st.image(img, use_container_width=True, caption=f"#{idx + 1}")
                else:
                    st.markdown("🖼️ *No disponible*")
            except Exception:
                st.markdown("🖼️ *Error cargando imagen*")

        with col_info:
            # Score badge
            score_pct = int(score * 100)
            st.markdown(
                f'<div class="score-badge">Similitud: {score_pct}%</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**ID:** `{image_id[:8]}...`")
            st.markdown(f"**Archivo:** `{filename}`")

            if description:
                with st.expander("📝 Descripción VQA"):
                    st.markdown(description[:400])

            if tags:
                tags_html = "".join(f'<span class="tag-badge">{t}</span>' for t in tags)
                st.markdown(tags_html, unsafe_allow_html=True)

            # VQA interactivo
            with st.expander("❓ Hacer pregunta a esta imagen"):
                q = st.text_input(
                    "Pregunta (en inglés para mejor precisión):",
                    key=f"vqa_q_{image_id}",
                    placeholder="What color is the car?",
                )
                if st.button("Preguntar", key=f"vqa_btn_{image_id}"):
                    with st.spinner("BLIP respondiendo..."):
                        try:
                            r = requests.post(
                                f"{api_url}/vqa",
                                json={"image_id": image_id, "question": q},
                                timeout=30,
                            )
                            if r.status_code == 200:
                                ans = r.json().get("answer", "")
                                st.success(f"**R:** {ans}")
                            else:
                                st.error("Error en VQA")
                        except Exception as e:
                            st.error(f"Error: {e}")

        st.markdown("---")


def render_search_page():
    st.title("🔍 Buscar Imágenes")
    st.markdown("*Describe lo que recuerdas haber visto en la imagen*")

    api_url = _get_api_url()
    top_k = st.session_state.get("top_k", 5)
    threshold = st.session_state.get("threshold", 0.20)

    # ── Modo de búsqueda ──────────────────────────────────────────────────────
    tab_text, tab_image = st.tabs(["📝 Por descripción", "🖼️ Por imagen similar"])

    # ── Búsqueda por texto ────────────────────────────────────────────────────
    with tab_text:
        col_query, col_opts = st.columns([3, 1])

        with col_query:
            query = st.text_area(
                "¿Qué recuerdas de la imagen?",
                placeholder="Ej: una foto de un perro jugando en un parque con árboles verdes al fondo...",
                height=100,
            )

        with col_opts:
            use_agent = st.checkbox("🤖 Usar agente Ollama", value=False)
            st.markdown(
                "*El agente razona y puede hacer preguntas VQA adicionales*"
                if use_agent
                else "*Búsqueda directa CLIP, más rápida*"
            )

        search_btn = st.button("🔍 Buscar", type="primary", use_container_width=True)

        if search_btn and query.strip():
            with st.spinner("Buscando..." if not use_agent else "El agente está buscando..."):
                try:
                    resp = requests.post(
                        f"{api_url}/search",
                        json={
                            "query": query,
                            "top_k": top_k,
                            "threshold": threshold,
                            "use_agent": use_agent,
                        },
                        timeout=120,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        results = data.get("search_results", [])
                        response_text = data.get("response", "")
                        tool_calls = data.get("tool_calls", [])
                        iterations = data.get("iterations", 1)

                        # Respuesta del agente
                        if use_agent and response_text:
                            st.markdown("### 🤖 Respuesta del Agente")
                            st.markdown(
                                f'<div class="agent-response">{response_text}</div>',
                                unsafe_allow_html=True,
                            )

                            # Herramientas usadas
                            if tool_calls:
                                tools_html = "".join(
                                    f'<span class="tool-pill">🔧 {tc["tool"]}</span>'
                                    for tc in tool_calls
                                )
                                st.markdown(
                                    f"**Herramientas usadas ({iterations} iter.):** {tools_html}",
                                    unsafe_allow_html=True,
                                )

                        # Resultados
                        st.markdown(f"### 📊 Resultados ({len(results)})")
                        if results:
                            for i, result in enumerate(results):
                                _render_result_card(result, api_url, i)
                        else:
                            st.info("No se encontraron imágenes que coincidan. Prueba con otra descripción.")

                    elif resp.status_code == 503:
                        st.error("⚠️ Ollama no está disponible. Verifique que esté corriendo.")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")

                except requests.exceptions.ConnectionError:
                    st.error(f"❌ No se puede conectar a la API en {api_url}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Búsqueda por imagen ───────────────────────────────────────────────────
    with tab_image:
        st.markdown("Sube una imagen de referencia para encontrar imágenes visualmente similares.")
        uploaded = st.file_uploader(
            "Imagen de referencia",
            type=["jpg", "jpeg", "png", "webp"],
            key="search_image_upload",
        )

        if uploaded:
            col_prev, col_res = st.columns([1, 2])
            with col_prev:
                st.image(uploaded, caption="Imagen de consulta", use_container_width=True)

            if st.button("🔍 Buscar imágenes similares", type="primary"):
                with st.spinner("Comparando con el repositorio..."):
                    try:
                        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                        resp = requests.post(
                            f"{api_url}/search/upload",
                            files=files,
                            data={"top_k": top_k},
                            timeout=60,
                        )

                        if resp.status_code == 200:
                            data = resp.json()
                            results = data.get("results", [])
                            with col_res:
                                st.markdown(f"**{len(results)} resultados encontrados:**")
                                for i, result in enumerate(results):
                                    _render_result_card(result, api_url, i)
                        else:
                            st.error(f"Error {resp.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ No se puede conectar a la API en {api_url}")
                    except Exception as e:
                        st.error(f"Error: {e}")
