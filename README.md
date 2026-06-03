# 🔍 Vision MCP Agent

Agente de búsqueda de imágenes por memoria visual usando CLIP, BLIP VQA, Ollama, FastAPI y Streamlit.

Describe lo que recuerdas haber visto en una imagen y el sistema la encuentra por similitud semántica usando embeddings CLIP.

El "MCP Client" está en agent/agent.py

 El proyecto tiene dos componentes MCP separados:

 ### 1. Cliente MCP → agent/agent.py (clase VisionAgent)

 El cliente MCP está integrado directamente en el agente. La clase VisionAgent en agent/agent.py actúa como       
 cliente MCP de la siguiente manera:

 - Define las herramientas disponibles (clip_search, vqa_query, list_repository, get_image_info) en la variable TOOLS.
 - Implementa un loop: envía el prompt + herramientas a Ollama → Ollama decide qué herramienta llamar → el agente ejecuta el handler → devuelve el    
   resultado a Ollama → repite hasta obtener respuesta final.
 - Los handlers (_build_handlers()) no se conectan al MCP server — llaman directamente al repositorio y modelos (CLIP, BLIP).

 ### 2. Servidor MCP → mcp_server/server.py

 Es un servidor MCP independiente que expone las mismas herramientas usando el protocolo MCP estándar (via stdio). Está pensado para que clientes MCP 
 externos (como Claude Desktop) se conecten a él.

 ### Resumen de la arquitectura:

 ```
   Usuario → API (api/main.py) → VisionAgent (agent/agent.py)
                                      ├── Ollama (LLM)
                                      └── Handlers directos → Repository / CLIP / BLIP

   [Externo] → mcp_server/server.py (stdio MCP server)
                   └── Herramientas expuestas via protocolo MCP estándar
 ```

 En resumen: el cliente MCP no está en un archivo separado; está embebido en agent/agent.py como parte de la clase VisionAgent, que hace de puente    
 entre Ollama y las herramientas de búsqueda visual.
---

## 🏗️ Arquitectura

```
vision-mcp-agent/
├── agent/                  # Agente Ollama + MCP
│   └── agent.py            # Loop LLM → tools → LLM
├── api/                    # FastAPI
│   ├── main.py             # Endpoints REST
│   └── schemas.py          # Modelos Pydantic
├── config/
│   └── settings.py         # Toda la configuración
├── mcp_server/             # Servidor MCP
│   └── server.py
├── models/
│   ├── clip_model.py       # Embeddings CLIP
│   └── blip_model.py       # VQA con BLIP
├── repository/
│   ├── manager.py          # CRUD + búsqueda
│   └── storage/            # Imágenes, embeddings, index.json
├── ui/
│   ├── app.py              # Streamlit (entry point)
│   └── pages/
│       ├── search.py       # 🔍 Búsqueda
│       ├── repository.py   # 🗂️ Repositorio
│       ├── chat.py         # 💬 Chat con agente
│       └── status.py       # ℹ️ Health check
├── .env                    # Variables de entorno
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerrequisitos

### 1. Instalar Ollama (solo si usás el agente con LLM)

Descargalo de [ollama.com](https://ollama.com) e instálalo.

Luego bajá el modelo que quieras usar:

```bash
# Modelo recomendado (multimodal, español/inglés)
ollama pull qwen2.5:7b

# Alternativas:
ollama pull llama3.2-vision   # Meta, 11B
ollama pull gemma3:12b        # Google, 12B
```

Verificá que esté corriendo:

```bash
ollama list
```

### 2. Docker

Instalá [Docker Desktop](https://www.docker.com/products/docker-desktop/) para Windows.

---

## 🚀 Ejecutar el proyecto

### Opción A — Con Docker (recomendado)

```bash
cd src/mcp/vision-mcp-agent

# Levantar API + Streamlit
docker-compose up --build -d

# Ver logs
docker-compose logs -f api
```

| Servicio | URL |
|---|---|
| **Streamlit UI** | http://localhost:8501 |
| **API FastAPI** | http://localhost:8001 |
| **API Docs** | http://localhost:8001/docs |

Para detener:

```bash
docker-compose down
```

### Opción B — Sin Docker

```bash
cd src/mcp/vision-mcp-agent
pip install -r requirements.txt

# Terminal 1 — API
uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2 — Streamlit
streamlit run ui/app.py --server.port 8501
```

---

## 🖼️ Indexar imágenes

Las imágenes deben indexarse para generar embeddings CLIP y descripciones BLIP. No basta con copiarlas a la carpeta.

### Desde la UI

1. Abrí http://localhost:8501
2. Sidebar → **🗂️ Repositorio**
3. Pestaña **➕ Agregar imagen**
4. Seleccioná una imagen y click en **📥 Indexar imagen**

### Batch — Indexar una carpeta completa

Usá el script `batch_index.py` que está en la **raíz del proyecto** (`visio-mcp-agent/`):

```bash
# Desde la raíz del proyecto
python batch_index.py ./ruta/a/tu/carpeta --api-url http://localhost:8001 --timeout 900

# Con tags para todas las imágenes
python batch_index.py ./mis_fotos --tags "vacaciones,playa,2024"

# Si la API está en Docker (hostname "api" no resuelve desde Windows)
python batch_index.py ./mis_fotos --api-url http://localhost:8001
```

> ⚠️ La primera indexación descarga los modelos CLIP y BLIP (~2 GB). Con `hf_cache` en Docker solo se descargan una vez. En CPU puede tardar 60-120s por imagen.

---

## 🔍 Buscar imágenes

1. Abres http://localhost:8501
2. Sidebar → **🔍 Buscar Imágenes**
3. Escribes una descripción en lenguaje natural

```
Ej: "una persona saltando en la orilla de una playa con olas"
```

4. Ajusta el umbral de similitud en el sidebar si no encontrás resultados

### Modos de búsqueda

| Modo | Descripción |
|---|---|
| **CLIP directo** (por defecto) | Compara embeddings CLIP. Rápido, no usa Ollama |
| **Agente Ollama** | El LLM razona, llama herramientas MCP, hace VQA. Requiere Ollama corriendo |

Para activar el agente: haz clic **🤖 Usar agente Ollama** en la página de búsqueda.

---

## 🔧 Cambiar el modelo de Ollama

Edita el archivo `.env`:

```env
# Cambiá esta línea por el modelo que prefieras
OLLAMA_MODEL=qwen2.5:7b
```

Modelos compatibles:

```env
OLLAMA_MODEL=llama3.2-vision
OLLAMA_MODEL=gemma3:12b
OLLAMA_MODEL=llava:13b
```

Reiniciá los contenedores después del cambio:

```bash
docker-compose down && docker-compose up -d
```

---

## 🎛️ Configuración

Toda la configuración está en `.env` y `config/settings.py`:

| Variable | Default | Descripción |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL de Ollama |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Modelo a usar |
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | Modelo CLIP (HuggingFace) |
| `BLIP_MODEL_NAME` | `Salesforce/blip-vqa-base` | Modelo BLIP (HuggingFace) |
| `CLIP_DEVICE` | `cpu` | `cpu` o `cuda` |
| `CLIP_SIMILARITY_THRESHOLD` | `0.20` | Umbral mínimo de similitud |
| `REPOSITORY_PATH` | `repository/storage/images` | Dónde se guardan las imágenes |
| `EMBEDDINGS_PATH` | `repository/storage/embeddings` | Dónde se guardan los embeddings |
| `API_PORT` | `8001` | Puerto de FastAPI |

---

## 🧠 ¿Cómo funciona?

1. **Indexado**: Al subir una imagen, CLIP genera un embedding vectorial (512 dimensiones) y BLIP responde 10 preguntas visuales. Ambos se almacenan.
2. **Búsqueda CLIP directa**: Tu descripción se convierte en embedding de texto. Se compara por similitud coseno con todos los embeddings de imagen.
3. **Agente Ollama + MCP** (opcional): El LLM recibe herramientas MCP (`clip_search`, `vqa_query`, `list_repository`). Decide cuáles usar, las invoca, y sintetiza la respuesta.

---

## 📂 Dónde se almacena todo

| Recurso | Ruta |
|---|---|
| Imágenes originales | `repository/storage/images/` |
| Embeddings CLIP (`.npy`) | `repository/storage/embeddings/` |
| Metadatos e índice | `repository/storage/index.json` |
| Caché modelos HuggingFace | `~/.cache/huggingface/` (volumen `hf_cache` en Docker) |

---

## ❗ Solución de problemas

**"Total imágenes indexadas: 0" a pesar de tener imágenes en la carpeta**
→ Las imágenes no fueron indexadas por la API. Ejecutá `batch_index.py`.

**"Ollama no está disponible"**
→ Ollama no está instalado o corriendo. Desmarca "Usar agente Ollama" para búsqueda CLIP directa.

**Timeout al indexar (error 500)**
→ Los modelos en CPU son lentos. Aumenta el timeout en el código o usa `--timeout 900` en batch_index.


**Para limpiar el volumen de las imágenes:**
 → 
 ```bash
   docker-compose -f src/mcp/vision-mcp-agent/docker-compose.yml down       
   docker volume rm vision-mcp-agent_repo_storage
   docker-compose -f src/mcp/vision-mcp-agent/docker-compose.yml up -d      
 ```