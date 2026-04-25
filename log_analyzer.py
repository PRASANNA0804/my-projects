"""
log_analyzer.py — Log File Parsing, Deduplication & RAG-Grounded Analysis
==========================================================================
Responsibilities:
  1. Detect whether an uploaded file is a log file
  2. Parse log lines to extract error/exception entries (multi-format)
  3. Deduplicate similar errors into groups
  4. For each group: retrieve relevant technical doc chunks via RAG
  5. Stream LLM analysis grounded ONLY in the retrieved documentation

Supported log formats:
  - Java log4j / logback  (timestamp LEVEL logger - message)
  - Spring Boot            (timestamp LEVEL pid --- [thread] logger : message)
  - Python logging module  (timestamp - name - LEVEL - message)
  - Syslog / journald      (Mon DD HH:MM:SS host proc[pid]: message)
  - JSON logs              (winston / bunyan / structlog)
  - Generic keyword match  (any line containing ERROR / SEVERE / EXCEPTION etc.)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Limits ────────────────────────────────────────────────────────────────────
MAX_LINES  = 50_000   # lines read from a large log file before truncating
MAX_GROUPS = 15       # max unique error groups to analyse (top by occurrence)
RAG_TOP_K  = 4        # chunks retrieved per error group

# ── LLM config (mirrors agent.py) ─────────────────────────────────────────────
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
LLM_MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── System prompt — strict doc-grounding ──────────────────────────────────────
LOG_ANALYSIS_SYSTEM_PROMPT = textwrap.dedent("""
    You are a technical root-cause analysis assistant.
    Your ONLY job is to explain errors found in log files using the documentation
    provided inside the <context> block.

    ABSOLUTE RULES — never break these:
    1. Answer ONLY from the provided <context>. Do not use any knowledge outside it,
       even if you know the answer from training data.
    2. If the context does not mention this error or component, respond with exactly:
       "The uploaded documentation does not cover this error. Please upload relevant
        technical documentation for [ErrorClass / component name]."
    3. Never say "typically", "generally", or "in most cases" — only cite what the
       documents actually say.
    4. Always reference the specific doc source, e.g.
       "According to [Source: api-guide.pdf]…"
    5. Structure every response as:

       ROOT CAUSE:
       [one sentence from the docs explaining why this error occurs]

       RECOMMENDED STEPS:
       1. [first step, grounded in the docs]
       2. [second step, grounded in the docs]
       …

       RELEVANT DOC SECTION:
       [source filename and a brief direct quote from the retrieved chunk]
""").strip()


# ── Compiled regex patterns ───────────────────────────────────────────────────
# 1. Java / Spring / Python structured log line
#    e.g. "2024-01-15 10:23:45,123 ERROR com.example.Svc - Connection refused"
_STRUCTURED = re.compile(
    r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[.,]\d{1,3}"
    r".*?\b(?P<level>SEVERE|ERROR|EXCEPTION|FATAL|CRITICAL|WARNING)\b"
    r".*?[-:]\s*(?P<msg>.+)",
    re.IGNORECASE,
)

# 2. Syslog / journald
#    e.g. "Jan 15 10:23:45 hostname myapp[1234]: ERROR something failed"
_SYSLOG = re.compile(
    r"^\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+\S+:\s*"
    r"(?P<level>SEVERE|ERROR|EXCEPTION|FATAL|CRITICAL|WARNING)\b"
    r"\s*:?\s*(?P<msg>.+)",
    re.IGNORECASE,
)

# 3. JSON log line (winston / bunyan / structlog)
#    e.g. {"level":"error","message":"Connection refused","timestamp":"..."}
_JSON_LINE = re.compile(
    r'^\s*\{.*"level"\s*:\s*"(?P<level>[^"]+)".*"message"\s*:\s*"(?P<msg>[^"]*)".*\}\s*$',
    re.IGNORECASE,
)

# 4. Keyword fallback — any line containing one of the target words
_KEYWORD = re.compile(
    r"\b(?P<level>SEVERE|ERROR|EXCEPTION|FATAL|CRITICAL|WARNING)\b",
    re.IGNORECASE,
)

# 5. Exception class name extractor
_EXC_CLASS = re.compile(
    r"(?:^|\s|:)([A-Z][a-zA-Z0-9]+(?:Exception|Error|Fault|Warning|Failure))"
)

# 6. Quick log-content detector (for .txt files)
_LOG_DETECTOR = re.compile(
    r"\b(SEVERE|ERROR|EXCEPTION|FATAL|CRITICAL|WARNING)\b",
    re.IGNORECASE,
)


# ── Data containers ───────────────────────────────────────────────────────────
@dataclass
class LogLine:
    """A single parsed log line that matched an error-level keyword."""
    level      : str
    message    : str
    raw_line   : str
    line_number: int


@dataclass
class ErrorGroup:
    """Deduplicated group of similar log lines."""
    key                   : str
    level                 : str
    sample_line           : str          # one representative raw line for display
    occurrences           : int
    representative_message: str          # normalised message used for RAG query
    exc_class             : Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_exc_class(text: str) -> Optional[str]:
    """Return the first Java/Python exception class name found in *text*, or None."""
    m = _EXC_CLASS.search(text)
    return m.group(1) if m else None


def _make_group_key(level: str, exc_class: Optional[str], message: str) -> str:
    """
    Build a stable deduplication key.
    - Strip all digits (line numbers, IDs change per run)
    - Collapse whitespace
    - Hash the first 120 normalised characters
    """
    normalised = re.sub(r"\d+", "N", message)
    normalised = re.sub(r"\s+", " ", normalised).strip()
    digest = hashlib.md5(normalised[:120].encode()).hexdigest()[:8]
    cls = exc_class or "GENERIC"
    return f"{level.upper()}::{cls}::{digest}"


# ── Public: file detection ─────────────────────────────────────────────────────
def is_log_file(filename: str, content_preview: str) -> bool:
    """
    Return True if the file should be treated as a log file.
    - Any file with a .log extension
    - A .txt file whose first 4 KB contains log-level keywords
    """
    ext = Path(filename).suffix.lower()
    if ext == ".log":
        return True
    if ext == ".txt":
        return bool(_LOG_DETECTOR.search(content_preview))
    return False


# ── Public: parsing ───────────────────────────────────────────────────────────
def parse_log_lines(content: str, max_lines: int = MAX_LINES) -> List[LogLine]:
    """
    Extract error-level log lines from *content* using a four-pass strategy:
      Pass 1 — structured (Java/Spring/Python timestamped lines)
      Pass 2 — syslog / journald
      Pass 3 — JSON lines (winston / bunyan)
      Pass 4 — keyword fallback (any remaining unmatched lines)
    """
    results: List[LogLine] = []
    lines = content.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]

    for lineno, raw in enumerate(lines, start=1):
        raw = raw.rstrip()
        if not raw:
            continue

        matched = False

        # Pass 1: structured timestamp log
        m = _STRUCTURED.match(raw)
        if m:
            level = m.group("level").upper()
            msg   = m.group("msg").strip()
            results.append(LogLine(level=level, message=msg, raw_line=raw, line_number=lineno))
            matched = True

        # Pass 2: syslog
        if not matched:
            m = _SYSLOG.match(raw)
            if m:
                level = m.group("level").upper()
                msg   = m.group("msg").strip()
                results.append(LogLine(level=level, message=msg, raw_line=raw, line_number=lineno))
                matched = True

        # Pass 3: JSON line
        if not matched:
            m = _JSON_LINE.match(raw)
            if m:
                level = m.group("level").upper()
                # Normalise level names that don't exactly match our keywords
                if level in ("ERR",):
                    level = "ERROR"
                if level in ("WARN",):
                    level = "WARNING"
                if level not in ("SEVERE", "ERROR", "EXCEPTION", "FATAL", "CRITICAL", "WARNING"):
                    continue
                msg = m.group("msg").strip()
                results.append(LogLine(level=level, message=msg, raw_line=raw, line_number=lineno))
                matched = True

        # Pass 4: keyword fallback
        if not matched:
            km = _KEYWORD.search(raw)
            if km:
                level = km.group("level").upper()
                # Use the text after the keyword as the message
                after = raw[km.end():].lstrip(": ").strip()
                msg   = after if after else raw.strip()
                results.append(LogLine(level=level, message=msg, raw_line=raw, line_number=lineno))

    return results


# ── Public: deduplication ─────────────────────────────────────────────────────
def deduplicate_errors(
    lines: List[LogLine],
    max_groups: int = MAX_GROUPS,
) -> List[ErrorGroup]:
    """
    Group similar log lines by (level, exception_class, normalised_message_hash).
    Returns at most *max_groups* groups, sorted by occurrence count descending.
    """
    groups: Dict[str, ErrorGroup] = {}

    for ll in lines:
        exc_class = _extract_exc_class(ll.raw_line) or _extract_exc_class(ll.message)
        key = _make_group_key(ll.level, exc_class, ll.message)

        if key in groups:
            groups[key].occurrences += 1
        else:
            groups[key] = ErrorGroup(
                key                    = key,
                level                  = ll.level,
                sample_line            = ll.raw_line,
                occurrences            = 1,
                representative_message = ll.message,
                exc_class              = exc_class,
            )

    # Sort by occurrence count (most common errors first)
    sorted_groups = sorted(groups.values(), key=lambda g: g.occurrences, reverse=True)
    return sorted_groups[:max_groups]


# ── Public: async streaming analysis ─────────────────────────────────────────
async def analyze_log_stream(
    content  : str,
    filename : str,
    async_client,          # AsyncAzureOpenAI — passed in from main.py to avoid circular import
) -> AsyncGenerator[dict, None]:
    """
    Async generator that yields SSE event dicts.

    Event types (in order):
      {"type": "no_docs"}                          — vector store is empty, stop
      {"type": "start", "filename", "total_errors", "skipped"}
      {"type": "analyzing", "group_index", "total_groups", "error_key",
                            "level", "sample_line", "occurrences"}
      {"type": "sources", "group_index", "sources": [...]}
      {"type": "token", "group_index", "token"}
      {"type": "group_done", "group_index", "no_docs_for_group"}
      {"type": "done", "groups_analyzed", "total_errors_found"}
      {"type": "error", "message"}
    """
    from retriever import retrieve, format_context

    # ── Parse ───────────────────────────────────────────────────────────────
    all_lines = parse_log_lines(content, max_lines=MAX_LINES)
    groups    = deduplicate_errors(all_lines, max_groups=MAX_GROUPS)

    total_occurrences = sum(g.occurrences for g in groups)
    skipped_lines     = len(all_lines) - total_occurrences  # lines that parsed but fell outside top groups

    # ── Guard: check vector store has documents ──────────────────────────────
    probe = retrieve("error exception", top_k=1)
    if not probe:
        yield {"type": "no_docs"}
        return

    # ── Nothing error-level in this log ─────────────────────────────────────
    if not groups:
        yield {
            "type"         : "start",
            "filename"     : filename,
            "total_errors" : 0,
            "skipped"      : len(all_lines),
        }
        yield {
            "type"           : "done",
            "groups_analyzed": 0,
            "total_errors_found": 0,
        }
        return

    yield {
        "type"        : "start",
        "filename"    : filename,
        "total_errors": len(groups),
        "skipped"     : skipped_lines,
    }

    # ── Analyse each error group ─────────────────────────────────────────────
    for i, group in enumerate(groups):
        yield {
            "type"       : "analyzing",
            "group_index": i,
            "total_groups": len(groups),
            "error_key"  : group.key,
            "level"      : group.level,
            "sample_line": group.sample_line[:300],
            "occurrences": group.occurrences,
        }

        # Build a rich RAG query: exception class + representative message
        exc_prefix = group.exc_class + " " if group.exc_class else ""
        query = f"{exc_prefix}{group.level} {group.representative_message[:300]}"

        chunks = retrieve(query, top_k=RAG_TOP_K)

        yield {
            "type"       : "sources",
            "group_index": i,
            "sources"    : [
                {
                    "source"   : c.source,
                    "extension": c.metadata.get("extension", ""),
                    "relevance": c.relevance,
                    "preview"  : c.text[:120].replace("\n", " "),
                }
                for c in chunks
            ],
        }

        no_docs_for_group = len(chunks) == 0

        if not no_docs_for_group:
            context  = format_context(chunks)
            user_msg = (
                f"<context>\n{context}\n</context>\n\n"
                f"Log error to analyse:\n{group.sample_line}\n\n"
                f"Error message: {group.representative_message}"
            )

            try:
                stream = await async_client.chat.completions.create(
                    model      = AZURE_DEPLOYMENT,
                    stream     = True,
                    temperature= 0.1,          # very low — factual, doc-grounded
                    max_tokens = LLM_MAX_TOKENS,
                    messages   = [
                        {"role": "system", "content": LOG_ANALYSIS_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                )
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    token = chunk.choices[0].delta.content
                    if token:
                        yield {"type": "token", "group_index": i, "token": token}
            except Exception as exc:
                yield {
                    "type"       : "error",
                    "message"    : f"LLM call failed for group {i}: {exc}",
                }

        yield {
            "type"             : "group_done",
            "group_index"      : i,
            "no_docs_for_group": no_docs_for_group,
        }

    yield {
        "type"              : "done",
        "groups_analyzed"   : len(groups),
        "total_errors_found": total_occurrences,
    }


