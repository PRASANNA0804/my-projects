"""
main.py — FastAPI REST Backend (Azure OpenAI + Multi-format)
=============================================================
"""
import os, sys, time, traceback, tempfile, json
from pathlib import Path
from typing import List, Optional

import io
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

from openai import AsyncAzureOpenAI
from agent    import get_agent, get_chat_agent
from ingest   import ingest_document, VECTOR_DB_PATH, COLLECTION_NAME
from document_loader import SUPPORTED_EXTENSIONS, is_supported

app = FastAPI(
    title      ="RAG Agent API — Azure OpenAI",
    description="Multi-format RAG system powered by Azure OpenAI gpt-4o-mini",
    version    ="2.0.0",
    docs_url   ="/api/docs",
    redoc_url  ="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins    =["*"],
    allow_credentials=True,
    allow_methods    =["*"],
    allow_headers    =["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k   : int = Field(default=5, ge=1, le=20)

class SourceChunk(BaseModel):
    source    : str
    extension : str
    relevance : float
    preview   : str

class ChatResponse(BaseModel):
    answer     : str
    sources    : List[SourceChunk]
    model      : str
    latency_ms : int
    tokens_used: int

class ConverseRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history : List[dict] = Field(default_factory=list)
    web_context: Optional[str] = None
    persona: Optional[str] = "aria"

class ConverseResponse(BaseModel):
    answer     : str
    model      : str
    tokens_used: int

class IngestResponse(BaseModel):
    status       : str
    chunks_stored: int
    filename     : str
    file_type    : str

class HealthResponse(BaseModel):
    status         : str
    vector_store   : str
    chunk_count    : int
    azure_endpoint : str
    deployment     : str

class StatsResponse(BaseModel):
    total_chunks      : int
    sources           : List[dict]
    model             : str
    supported_formats : List[str]

class DeleteResponse(BaseModel):
    status : str
    deleted: str

class IngestUrlRequest(BaseModel):
    url      : str
    depth    : int = Field(default=2, ge=1, le=3)
    max_pages: int = Field(default=10, ge=1, le=50)


# ── Vector store helpers ───────────────────────────────────────────────────────
def _get_collection():
    client = chromadb.PersistentClient(
        path    =str(VECTOR_DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(COLLECTION_NAME)

def _collection_stats():
    try:
        col   = _get_collection()
        count = col.count()
        if count > 0:
            metas = col.get(include=["metadatas"])["metadatas"] or []
            sources_map: dict[str, dict] = {}
            for m in metas:
                if m is None:
                    continue
                src = str(m.get("source", "unknown"))
                if src not in sources_map:
                    sources_map[src] = {"name": src, "extension": str(m.get("extension", "")), "chunks": 0}
                sources_map[src]["chunks"] += 1
            return count, list(sources_map.values())
        return 0, []
    except Exception:
        return 0, []


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health():
    count, _ = _collection_stats()
    return HealthResponse(
        status        ="ok",
        vector_store  ="chromadb",
        chunk_count   =count,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "not set"),
        deployment    =os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
    )

@app.get("/api/stats", response_model=StatsResponse, tags=["system"])
def stats():
    count, sources = _collection_stats()
    return StatsResponse(
        total_chunks     =count,
        sources          =sources,
        model            =os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        supported_formats=SUPPORTED_EXTENSIONS,
    )

@app.post("/api/chat", response_model=ChatResponse, tags=["rag"])
def chat(req: ChatRequest):
    try:
        agent = get_agent()
        t0    = time.perf_counter()
        resp  = agent.answer(req.question)
        ms    = int((time.perf_counter() - t0) * 1000)

        sources = [
            SourceChunk(
                source   =c.source,
                extension=c.metadata.get("extension", ""),
                relevance=c.relevance,
                preview  =c.text[:250].replace("\n", " "),
            )
            for c in resp.chunks
        ]

        return ChatResponse(
            answer     =resp.answer,
            sources    =sources,
            model      =resp.model,
            latency_ms =ms,
            tokens_used=resp.tokens_used,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/converse", response_model=ConverseResponse, tags=["chat"])
def converse(req: ConverseRequest):
    try:
        from typing import cast
        from openai.types.chat import ChatCompletionMessageParam
        agent   = get_chat_agent()
        history = cast(List[ChatCompletionMessageParam], [
            {"role": m["role"], "content": m["content"]}
            for m in req.history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ])
        # Inject live web results as a grounded context the model must use
        actual_message = req.message
        if req.web_context:
            actual_message = (
                "🔴 LIVE DATA INJECTED — you have real-time web search results right now. "
                "Do NOT say you lack real-time access. Use the results below to answer.\n\n"
                + req.web_context
                + "\n\n---\nUser question: " + req.message
            )
        resp = agent.chat(message=actual_message, history=history or None, persona=req.persona or "aria")
        return ConverseResponse(
            answer     =resp.answer,
            model      =resp.model,
            tokens_used=resp.tokens_used,
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest_file(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if not is_supported(filename):
        raise HTTPException(
            status_code=400,
            detail=f"Format '{suffix}' not supported. Accepted: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    tmp = Path(tempfile.gettempdir()) / filename
    tmp.write_bytes(await file.read())
    try:
        n = ingest_document(str(tmp))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp.unlink(missing_ok=True)

    return IngestResponse(
        status       ="success",
        chunks_stored=n,
        filename     =filename,
        file_type    =suffix,
    )

@app.delete("/api/document/{filename}", response_model=DeleteResponse, tags=["rag"])
def delete_document(filename: str):
    """Remove all chunks belonging to a specific source file."""
    try:
        col     = _get_collection()
        results = col.get(where={"source": filename})
        ids     = results.get("ids", [])
        if ids:
            col.delete(ids=ids)
        return DeleteResponse(status="ok", deleted=f"{len(ids)} chunks from '{filename}'")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.delete("/api/clear-all", tags=["rag"])
def clear_all():
    """Wipe the entire vector store."""
    try:
        col = _get_collection()
        all_ids = col.get()["ids"]
        if all_ids:
            col.delete(ids=all_ids)
        return {"status": "ok", "message": "Vector store cleared"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/ingest-url", tags=["rag"])
async def ingest_url_endpoint(req: IngestUrlRequest):
    """Crawl a URL (optionally following links) and ingest all pages into ChromaDB.
    Streams Server-Sent Events with live progress per page."""
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="A valid http:// or https:// URL is required")

    from ingest import crawl_and_ingest

    async def generate():
        try:
            async for event in crawl_and_ingest(url, depth=req.depth, max_pages=req.max_pages):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            traceback.print_exc()
            yield f"data: {json.dumps({'type':'error','url':url,'message':str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/web-search", tags=["chat"])
async def web_search(q: str, max_results: int = 5):
    """Real-time web search for ARIA chat — powers live data queries."""
    try:
        from ddgs import DDGS
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Web search not available. Run: pip install ddgs"
        )
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(q, max_results=max_results))
        if not results:
            return {"query": q, "results": [], "warning": "No results returned"}
        return {"query": q, "results": results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

# ── Async client for streaming ─────────────────────────────────────────────────
_async_client: AsyncAzureOpenAI | None = None

def _get_async_client() -> AsyncAzureOpenAI:
    global _async_client
    if _async_client is None:
        ep  = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        key = os.getenv("AZURE_OPENAI_API_KEY",  "")
        ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        if not ep or not key:
            raise RuntimeError("Azure OpenAI credentials not set in .env")
        _async_client = AsyncAzureOpenAI(azure_endpoint=ep, api_key=key, api_version=ver)
    return _async_client


@app.post("/api/chat/stream", tags=["rag"])
async def chat_stream(req: ChatRequest):
    """Streaming RAG — returns Server-Sent Events with token deltas."""
    from retriever import retrieve, format_context
    from agent import (RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE,
                       AZURE_DEPLOYMENT, LLM_TEMPERATURE, LLM_MAX_TOKENS)
    client = _get_async_client()

    async def generate():
        try:
            chunks = retrieve(req.question, top_k=req.top_k)
            if not chunks:
                yield f"data: {json.dumps({'type':'no_docs'})}\n\n"
                return

            sources = [
                {"source": c.source, "extension": c.metadata.get("extension",""),
                 "relevance": c.relevance, "preview": c.text[:250].replace("\n"," ")}
                for c in chunks
            ]
            yield f"data: {json.dumps({'type':'sources','sources':sources})}\n\n"

            context  = format_context(chunks)
            user_msg = RAG_USER_TEMPLATE.format(context=context, question=req.question)

            stream = await client.chat.completions.create(
                model=AZURE_DEPLOYMENT, temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS, stream=True,
                messages=[{"role":"system","content":RAG_SYSTEM_PROMPT},
                          {"role":"user",  "content":user_msg}],
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type':'token','token':delta})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        except Exception as exc:
            traceback.print_exc()
            yield f"data: {json.dumps({'type':'error','message':str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@app.post("/api/converse/stream", tags=["chat"])
async def converse_stream(req: ConverseRequest):
    """Streaming Chat (ARIA) — returns Server-Sent Events with token deltas."""
    from typing import cast
    from openai.types.chat import ChatCompletionMessageParam
    from agent import (PERSONA_PROMPTS, CHAT_SYSTEM_PROMPT,
                       AZURE_DEPLOYMENT, CHAT_TEMPERATURE, LLM_MAX_TOKENS)
    client = _get_async_client()

    async def generate():
        try:
            history = cast(List[ChatCompletionMessageParam], [
                {"role": m["role"], "content": m["content"]}
                for m in req.history
                if m.get("role") in ("user","assistant") and m.get("content")
            ])
            actual_message = req.message
            if req.web_context:
                actual_message = (
                    "🔴 LIVE DATA INJECTED — you have real-time web search results right now. "
                    "Do NOT say you lack real-time access. Use the results below to answer.\n\n"
                    + req.web_context + "\n\n---\nUser question: " + req.message
                )
            system_prompt = PERSONA_PROMPTS.get(req.persona or "aria", CHAT_SYSTEM_PROMPT)
            messages: list[ChatCompletionMessageParam] = [{"role":"system","content":system_prompt}]
            if history:
                messages.extend(history[-20:])
            messages.append({"role":"user","content":actual_message})

            stream = await client.chat.completions.create(
                model=AZURE_DEPLOYMENT, temperature=CHAT_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS, stream=True, messages=messages,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type':'token','token':delta})}\n\n"
            yield f"data: {json.dumps({'type':'done'})}\n\n"
        except Exception as exc:
            traceback.print_exc()
            yield f"data: {json.dumps({'type':'error','message':str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
