# streamlit_app/pages/repository.py
"""Página de gestión del repositorio."""
import io
import requests
import streamlit as st
from PIL import Image


def _get_api_url() -> str:
    return st.session_state.get("api_url", "http://api:8001")


def render_repository_page():
    st.title("Repositorio de Imágenes")
    api_url = _get_api_url()

    tab_list, tab_add = st.tabs(["Ver imágenes", "➕ Agregar imagen"])

    # ── Listado ───────────────────────────────────────────────────────────────
    with tab_list:
        col_refresh, col_count = st.columns([1, 3])
        with col_refresh:
            if st.button("🔄 Actualizar"):
                st.rerun()

        try:
            resp = requests.get(f"{api_url}/repository?limit=50", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                total = data.get("total", 0)
                images = data.get("images", [])

                with col_count:
                    st.metric("Total imágenes indexadas", total)

                if not images:
                    st.info("El repositorio está vacío. Agrega imágenes en la pestaña '➕ Agregar imagen'.")
                    

                # Grid de imágenes
                cols_per_row = 3
                for i in range(0, len(images), cols_per_row):
                    row_imgs = images[i: i + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, img_meta in zip(cols, row_imgs):
                        with col:
                            image_id = img_meta["image_id"]
                            try:
                                img_resp = requests.get(f"{api_url}/images/{image_id}", timeout=8)
                                if img_resp.status_code == 200:
                                    img = Image.open(io.BytesIO(img_resp.content))
                                    st.image(img, use_container_width=True)
                            except Exception:
                                st.markdown("🖼️ *Error*")

                            st.markdown(f"**`{image_id[:8]}...`**")
                            dims = f"{img_meta.get('width', '?')}×{img_meta.get('height', '?')}"
                            st.caption(f"{img_meta['filename']} | {dims}")

                            tags = img_meta.get("tags", [])
                            if tags:
                                st.caption(f"Tags: {', '.join(tags)}")

                            if st.button("🗑️ Eliminar", key=f"del_{image_id}", type="secondary"):
                                del_resp = requests.delete(f"{api_url}/repository/{image_id}", timeout=10)
                                if del_resp.status_code == 200:
                                    st.success("Eliminada")
                                    st.rerun()
                                else:
                                    st.error("Error eliminando")

        except requests.exceptions.ConnectionError:
            st.error(f"❌ No se puede conectar a la API en {api_url}")
        except Exception as e:
            st.error(f"Error: {e}")

    # ── Agregar imagen ────────────────────────────────────────────────────────
    with tab_add:
        st.markdown("### Subir imagen al repositorio")
        st.info("Al subir una imagen, se generarán automáticamente:\n- **Embedding CLIP** para búsqueda semántica\n- **Descripción VQA BLIP** con 10 preguntas visuales")

        uploaded_file = st.file_uploader(
            "Selecciona una imagen",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="repo_upload",
        )

        tags_input = st.text_input(
            "Tags (opcional, separados por coma)",
            placeholder="naturaleza, perro, exterior",
        )

        if uploaded_file:
            col_prev, col_info = st.columns([1, 1])
            with col_prev:
                st.image(uploaded_file, caption="Vista previa", use_container_width=True)
            with col_info:
                st.markdown(f"**Archivo:** {uploaded_file.name}")
                st.markdown(f"**Tipo:** {uploaded_file.type}")
                st.markdown(f"**Tamaño:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")

            if st.button("📥 Indexar imagen", type="primary", use_container_width=True):
                with st.spinner("Generando embeddings CLIP y descripción VQA BLIP... (puede tardar 30-60s)"):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {"tags": tags_input}
                        resp = requests.post(
                            f"{api_url}/repository/add",
                            files=files,
                            data=data,
                            timeout=300,
                        )

                        if resp.status_code == 200:
                            result = resp.json()
                            st.success(f"✅ Imagen indexada correctamente!")
                            st.json({
                                "image_id": result["image_id"],
                                "filename": result["filename"],
                                "dimensions": result["dimensions"],
                                "tags": result["tags"],
                                "description_preview": result.get("searchable_text_preview", ""),
                            })
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text}")
                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ No se puede conectar a la API en {api_url}")
                    except Exception as e:
                        st.error(f"Error: {e}")
