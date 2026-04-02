"""
ingest.py — Multi-Format Document Ingestion Pipeline
=====================================================
Loads ANY supported document type, extracts text, chunks it,
embeds it, and persists to ChromaDB.

Supported formats: .txt, .md, .pdf, .docx, .xlsx, .pptx, .csv, .json, .rtf
"""

import asyncio
import glob
import os
from pathlib import Path
from typing import List, Dict, Optional, cast
from urllib.parse import urlparse, urljoin, urldefrag

from openai import AzureOpenAI
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Embeddings, Metadata
from dotenv import load_dotenv

from document_loader import extract_text, is_supported, SUPPORTED_EXTENSIONS

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
DOCS_FOLDER     = Path(__file__).parent.parent / "docs"
VECTOR_DB_PATH  = Path(__file__).parent.parent / "vector_store"
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE      = 600   # chars per chunk (slightly larger for richer context)
CHUNK_OVERLAP   = 80    # overlap chars
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")


# ── Step 1: Load documents using the universal loader ─────────────────────────
def load_documents(folder: Path) -> List[Dict[str, str]]:
    documents = []
    # Glob ALL files, filter by supported extension
    all_files = glob.glob(str(folder / "**/*"), recursive=True)
    supported = [f for f in all_files if is_supported(f) and Path(f).is_file()]

    if not supported:
        print(f"⚠  No supported files found in {folder}")
        print(f"   Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        return documents

    for path in supported:
        try:
            text = extract_text(path)
            if text and text.strip():
                documents.append({
                    "filename"  : Path(path).name,
                    "extension" : Path(path).suffix.lower(),
                    "content"   : text.strip(),
                })
                print(f"   ✔ Loaded [{Path(path).suffix.upper()}]: {Path(path).name}  ({len(text):,} chars)")
        except Exception as e:
            print(f"   ✖ Failed: {Path(path).name} — {e}")

    return documents


# ── Step 2: Chunk text with overlap ───────────────────────────────────────────
def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character chunks."""
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 30]  # drop tiny trailing chunks


# ── Step 3: Azure embedding client ────────────────────────────────────────────
def get_embedding_client() -> AzureOpenAI:
    ep  = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    key = os.getenv("AZURE_OPENAI_API_KEY", "")
    ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    if not ep or not key:
        raise RuntimeError("Azure OpenAI credentials not set in .env")
    print(f"   Using Azure embedding model: {EMBEDDING_DEPLOYMENT} …")
    return AzureOpenAI(azure_endpoint=ep, api_key=key, api_version=ver)


def embed_texts(client: AzureOpenAI, texts: List[str]) -> Embeddings:
    """Embed a list of texts using Azure OpenAI — batched in groups of 100."""
    all_embeddings: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=EMBEDDING_DEPLOYMENT)
        all_embeddings.extend([item.embedding for item in response.data])
    return cast(Embeddings, all_embeddings)


def get_vector_store() -> chromadb.Collection:
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path    =str(VECTOR_DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name    =COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Main entry-point ───────────────────────────────────────────────────────────
def ingest_document(file_path: Optional[str] = None) -> int:
    """
    Ingest a single file OR everything in DOCS_FOLDER.
    Returns number of chunks stored.
    """
    print("\n🚀  Starting ingestion …")

    if file_path:
        path = Path(file_path)
        if not is_supported(path):
            raise ValueError(f"Format '{path.suffix}' not supported. Use: {', '.join(SUPPORTED_EXTENSIONS)}")
        try:
            content = extract_text(path)
        except Exception as e:
            raise RuntimeError(f"Could not read {path.name}: {e}")
        documents = [{"filename": path.name, "extension": path.suffix.lower(), "content": content.strip()}]
        print(f"   File     : {path.name} ({path.suffix.upper()})")
    else:
        DOCS_FOLDER.mkdir(parents=True, exist_ok=True)
        documents = load_documents(DOCS_FOLDER)

    if not documents:
        print("❌  Nothing to ingest.")
        return 0

    embed_client = get_embedding_client()
    collection   = get_vector_store()
    total        = 0

    for doc in documents:
        chunks = split_into_chunks(doc["content"])
        if not chunks:
            continue

        ids        = [f"{doc['filename']}__chunk_{i}" for i in range(len(chunks))]
        embeddings = embed_texts(embed_client, chunks)
        metadatas = cast(List[Metadata], [
            {
                "source"     : doc["filename"],
                "extension"  : doc["extension"],
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ])

        collection.upsert(
            ids       =ids,
            documents =chunks,
            embeddings=embeddings,
            metadatas =metadatas,
        )
        total += len(chunks)
        print(f"   ✔ Stored {len(chunks)} chunks ← {doc['filename']}")

    print(f"\n✅  Done — {total} total chunks in vector store.\n")
    return total


async def crawl_and_ingest(url: str, depth: int = 2, max_pages: int = 10):
    """
    Async generator that crawls `url` up to `depth` levels deep (same domain only),
    ingests each page into ChromaDB, and yields progress dicts.

    depth=1 → only the given URL
    depth=2 → given URL + all links found on it
    depth=3 → two levels deep

    Yields:
      {"type": "progress", "page": int, "url": str, "chunks": int}
      {"type": "skip",     "url": str, "reason": str}
      {"type": "error",    "url": str, "message": str}
      {"type": "done",     "pages_ingested": int, "total_chunks": int}
    """
    try:
        import httpx
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]
        import trafilatura              # type: ignore[import-untyped]
    except ImportError as exc:
        yield {"type": "error", "url": url,
               "message": f"Missing dependency: {exc}. Run: pip install httpx beautifulsoup4 trafilatura"}
        return

    depth     = max(1, min(depth, 3))
    max_pages = max(1, min(max_pages, 50))

    parsed_start = urlparse(url)
    base_domain  = parsed_start.netloc

    # Sites that cannot be crawled (login walls, JS challenges, bot blocks)
    BLOCKED_DOMAINS: dict[str, str] = {
        "twitter.com":        "Twitter/X (login required)",
        "x.com":              "Twitter/X (login required)",
        "linkedin.com":       "LinkedIn (login required)",
        "facebook.com":       "Facebook (login required)",
        "instagram.com":      "Instagram (login required)",
        "tiktok.com":         "TikTok (bot detection)",
        "reddit.com":         "Reddit (bot detection / API required)",
        "nytimes.com":        "New York Times (paywall)",
        "ft.com":             "Financial Times (paywall)",
        "wsj.com":            "Wall Street Journal (paywall)",
        "bloomberg.com":      "Bloomberg (paywall)",
        "medium.com":         "Medium (metered paywall)",
        "github.com":         "GitHub (use raw.githubusercontent.com for raw files)",
    }

    visited: set  = set()
    queue         = [(url, 0)]   # (url, current_depth)
    pages_ingested = 0
    total_chunks   = 0
    pages_skipped  = 0

    try:
        embed_client = get_embedding_client()
        collection   = get_vector_store()
    except Exception as exc:
        yield {"type": "error", "url": url, "message": f"Vector store init failed: {exc}"}
        return

    loop = asyncio.get_running_loop()

    async with httpx.AsyncClient(
        timeout          = httpx.Timeout(20.0, connect=8.0),
        follow_redirects = True,
        verify           = False,   # skip SSL verify — avoids cert errors on some sites
        headers          = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
    ) as client:
        while queue and pages_ingested < max_pages:
            current_url, current_depth = queue.pop(0)
            current_url, _ = urldefrag(current_url)

            if current_url in visited:
                continue
            visited.add(current_url)

            # ── Block known unsupported sites ──────────────────────────────────
            cur_domain = urlparse(current_url).netloc.lower().removeprefix("www.")
            blocked_reason = next(
                (reason for domain, reason in BLOCKED_DOMAINS.items()
                 if cur_domain == domain or cur_domain.endswith("." + domain)),
                None
            )
            if blocked_reason:
                pages_skipped += 1
                yield {"type": "blocked", "url": current_url, "reason": blocked_reason}
                continue

            html = ""  # ensures html is always bound before link-collection block

            # ── Wikipedia REST API (avoids 403 bot-blocking) ───────────────────
            wp = urlparse(current_url)
            is_wikipedia = wp.netloc.endswith("wikipedia.org") and wp.path.startswith("/wiki/")
            if is_wikipedia:
                page_title = wp.path.split("/wiki/", 1)[1]
                # REST v1 /page/summary — fast, always JSON, no bot blocks
                rest_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
                yield {"type": "debug", "url": current_url, "msg": "Using Wikipedia REST API…"}
                text = ""
                try:
                    rest_resp = await client.get(
                        rest_url,
                        headers={"Accept": "application/json; charset=utf-8;",
                                 "User-Agent": "ARIA-RAG/1.0 (educational RAG project)"}
                    )
                    if rest_resp.status_code == 200:
                        rest_data = rest_resp.json()
                        text = rest_data.get("extract", "") or ""
                        text = text.strip()
                except Exception as exc:
                    text = ""
                    yield {"type": "debug", "url": current_url,
                           "msg": f"REST summary failed ({exc}), trying action API…"}

                # Fallback: action API (full article text, not just intro)
                if not text or len(text) < 80:
                    try:
                        action_url = (
                            f"https://en.wikipedia.org/w/api.php"
                            f"?action=query&titles={page_title}"
                            f"&prop=extracts&explaintext=true&exlimit=1&format=json"
                            f"&origin=*"
                        )
                        action_resp = await client.get(
                            action_url,
                            headers={"User-Agent": "ARIA-RAG/1.0 (educational RAG project)"}
                        )
                        if action_resp.status_code == 200 and action_resp.text.strip():
                            action_data = action_resp.json()
                            for page_val in action_data.get("query", {}).get("pages", {}).values():
                                text += page_val.get("extract", "") or ""
                            text = text.strip()
                    except Exception as exc2:
                        pages_skipped += 1
                        yield {"type": "skip", "url": current_url,
                               "reason": f"Wikipedia API unavailable: {exc2}"}
                        continue

                if not text or len(text) < 80:
                    pages_skipped += 1
                    yield {"type": "skip", "url": current_url,
                           "reason": "Wikipedia returned no content (page may not exist)"}
                    continue
            else:
                # ── Regular fetch ──────────────────────────────────────────────
                yield {"type": "debug", "url": current_url, "msg": "Fetching…"}
                try:
                    resp = await client.get(current_url)
                except httpx.TimeoutException:
                    pages_skipped += 1
                    yield {"type": "skip", "url": current_url, "reason": "Timeout"}
                    continue
                except Exception as exc:
                    pages_skipped += 1
                    yield {"type": "skip", "url": current_url, "reason": str(exc)}
                    continue

                if resp.status_code != 200:
                    pages_skipped += 1
                    yield {"type": "skip", "url": current_url,
                           "reason": f"HTTP {resp.status_code}"}
                    continue

                ct = resp.headers.get("content-type", "").lower()
                if "text/html" not in ct:
                    pages_skipped += 1
                    yield {"type": "skip", "url": current_url,
                           "reason": f"Not HTML (content-type: {ct[:60] or 'none'})"}
                    continue

                html = resp.text

                # ── Extract clean text ─────────────────────────────────────────
                text = trafilatura.extract(
                    html,
                    include_links  = False,
                    include_images = False,
                    include_tables = True,
                )
                # Fallback: BeautifulSoup visible-text
                if not text or len(text.strip()) < 80:
                    try:
                        soup_fb = BeautifulSoup(html, "html.parser")
                        for tag in soup_fb(["script", "style", "nav", "footer", "header", "aside"]):
                            tag.decompose()
                        text = soup_fb.get_text(separator=" ", strip=True)
                    except Exception:
                        text = None
                if not text or len(text.strip()) < 80:
                    pages_skipped += 1
                    yield {"type": "skip", "url": current_url, "reason": "No content extracted"}
                    continue

            # ── Chunk + embed + store (sync calls run in executor) ─────────────
            try:
                chunks = split_into_chunks(text.strip())
                if chunks:
                    embeddings = await loop.run_in_executor(
                        None, embed_texts, embed_client, chunks
                    )
                    source_key = current_url[:200]
                    url_hash   = abs(hash(current_url)) % (10 ** 12)
                    ids        = [f"url__{url_hash}__{i}" for i in range(len(chunks))]
                    metadatas  = cast(List[Metadata], [
                        {"source": source_key, "extension": ".url",
                         "chunk_index": i, "url": current_url}
                        for i in range(len(chunks))
                    ])
                    collection.upsert(
                        ids        = ids,
                        documents  = chunks,
                        embeddings = embeddings,
                        metadatas  = metadatas,
                    )
                    pages_ingested += 1
                    total_chunks   += len(chunks)
                    yield {"type": "progress", "page": pages_ingested,
                           "url": current_url, "chunks": len(chunks)}
            except Exception as exc:
                yield {"type": "error", "url": current_url, "message": str(exc)}
                continue

            # ── Collect links for next depth level ─────────────────────────────
            if current_depth < depth - 1 and not is_wikipedia:
                try:
                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        href = str(a.get("href") or "").strip()
                        if not href or href.startswith(("#", "mailto:", "javascript:")):
                            continue
                        absolute, _ = urldefrag(urljoin(current_url, href))
                        ap = urlparse(absolute)
                        if (ap.netloc == base_domain
                                and ap.scheme in ("http", "https")
                                and absolute not in visited
                                and len(queue) < max_pages * 4):
                            queue.append((absolute, current_depth + 1))
                except Exception:
                    pass   # link extraction failure is non-fatal

    yield {"type": "done", "pages_ingested": pages_ingested,
           "total_chunks": total_chunks, "pages_skipped": pages_skipped}


if __name__ == "__main__":
    import sys
    ingest_document(sys.argv[1] if len(sys.argv) > 1 else None)
