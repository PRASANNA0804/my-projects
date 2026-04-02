# ⟡ RAG Intelligence — Azure AI Foundry Edition

> Multi-format RAG system powered by **Azure OpenAI gpt-4o-mini**, with a stunning Vaadin-inspired UI and support for 8+ document formats.

---

## Azure AI Foundry Setup

From your Azure AI Foundry deployment (gpt-4o-mini, Global Standard):

```bash
# .env
AZURE_OPENAI_ENDPOINT= YOu API Endpoint
AZURE_OPENAI_API_KEY= Your API Key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=Your API Version
```

---

## Supported Document Formats

| Format | Extension | Library |
|--------|-----------|---------|
| PDF | `.pdf` | PyMuPDF (fitz) |
| Word | `.docx` | python-docx |
| Excel | `.xlsx` / `.xls` | openpyxl |
| PowerPoint | `.pptx` | python-pptx |
| Plain Text | `.txt` / `.log` | chardet |
| Markdown | `.md` | built-in |
| CSV | `.csv` | built-in |
| JSON | `.json` / `.jsonl` | built-in |
| RTF | `.rtf` | striprtf |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Configure Azure
cp .env.example .env
# → Edit .env with your Azure credentials

# 3. Ingest sample document
python backend/ingest.py

# 4. Start API
python backend/main.py
# → http://localhost:8000/api/docs

# 5. Open UI
open frontend/index.html
```

---

## Project Structure

```
rag_agent/
├── backend/
│   ├── document_loader.py  ← Universal multi-format text extractor
│   ├── ingest.py           ← Chunk + embed + store pipeline
│   ├── retriever.py        ← Cosine similarity search
│   ├── agent.py            ← Azure OpenAI RAG agent
│   ├── main.py             ← FastAPI REST server
│   ├── watcher.py          ← File watcher / webhook trigger
│   ├── mcp_server.py       ← MCP server for Claude Desktop
│   └── requirements.txt
├── frontend/
│   └── index.html          ← Vaadin-powered enriched chat UI
├── docs/                   ← Drop your documents here
├── vector_store/           ← ChromaDB persists here
├── .env.example
└── README.md
```

---

## UI Features

- **Animated particle network** background
- **Live sidebar** — API latency, chunk stats, source file manager
- **Per-source delete** — remove individual documents from the KB
- **Format badges** — colour-coded by file type (PDF, DOCX, XLSX…)
- **Source citations** — click to expand preview from retrieved chunks
- **Token counter** — tracks Azure API usage in real time
- **Top-K selector** — adjust retrieval depth from the input bar
- **Drag & drop upload** — ingest any supported format instantly
- **Amber/gold design** — warm enterprise aesthetic on deep ink backgrounds

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness + Azure endpoint check |
| `GET` | `/api/stats` | Chunk count, sources, supported formats |
| `POST` | `/api/chat` | RAG chat with Azure OpenAI |
| `POST` | `/api/ingest` | Upload & ingest any supported file |
| `DELETE` | `/api/document/{name}` | Remove document chunks |
| `DELETE` | `/api/clear-all` | Wipe entire vector store |
