"""
agent.py — Dual-Mode Agent: RAG + Conversational (Azure OpenAI)
===============================================================
Two agents, one file:

  RAGAgent         — document-grounded answers with source citations
  ConversationalAgent — funny, friendly, all-knowing, multilingual chat
                        with full conversation history support
"""

from __future__ import annotations
import os, textwrap
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv
load_dotenv()

from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from retriever import retrieve, format_context, RetrievedChunk

# ── Azure config ───────────────────────────────────────────────────────────────
AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY",  "")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
LLM_TEMPERATURE   = float(os.getenv("LLM_TEMP",       "0.2"))
CHAT_TEMPERATURE  = float(os.getenv("CHAT_TEMP",      "0.85"))   # higher = more fun
LLM_MAX_TOKENS    = int(os.getenv("LLM_MAX_TOKENS",   "1024"))
RAG_TOP_K         = int(os.getenv("RAG_TOP_K",        "5"))


# ══════════════════════════════════════════════════════════════════════════════
# RAG AGENT — document-grounded
# ══════════════════════════════════════════════════════════════════════════════
RAG_SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent, precise document assistant.
    Answer questions using the provided <context> from the knowledge base.

    Rules:
    - Ground every answer in the provided context.
    - If the context contains related information (even partial), use it to construct the best possible answer.
    - Only say "The documents don't cover this — try Chat mode for general questions!" if the context has absolutely NO relevant information whatsoever.
    - Cite sources (e.g. "According to report.pdf…").
    - Use bullet points or numbered steps when helpful.
    - Mention the document type when relevant (e.g. "In the spreadsheet…").
    - Be concise but thorough.
""").strip()

RAG_USER_TEMPLATE = textwrap.dedent("""
    <context>
    {context}
    </context>

    Question: {question}

    Answer:
""").strip()


@dataclass
class AgentResponse:
    answer     : str
    question   : str
    sources    : List[str]            = field(default_factory=list)
    chunks     : List[RetrievedChunk] = field(default_factory=list)
    model      : str                  = AZURE_DEPLOYMENT
    tokens_used: int                  = 0
    mode       : str                  = "rag"

    @property
    def unique_sources(self) -> List[str]:
        return sorted(set(self.sources))


class RAGAgent:
    """Stateless RAG agent — each call is independent."""

    def __init__(self, client: AzureOpenAI) -> None:
        self._client = client

    def answer(self, question: str) -> AgentResponse:
        question = question.strip()
        if not question:
            return AgentResponse(answer="Please ask something!", question=question)

        chunks = retrieve(question, top_k=RAG_TOP_K)
        if not chunks:
            return AgentResponse(
                answer=(
                    "🔍 No matching documents found in the knowledge base.\n\n"
                    "Try uploading some files first, or switch to **Chat mode** "
                    "to ask me anything without needing documents!"
                ),
                question=question, mode="rag",
            )

        context  = format_context(chunks)
        user_msg = RAG_USER_TEMPLATE.format(context=context, question=question)

        resp = self._client.chat.completions.create(
            model      =AZURE_DEPLOYMENT,
            temperature=LLM_TEMPERATURE,
            max_tokens =LLM_MAX_TOKENS,
            messages   =[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )

        return AgentResponse(
            answer     =(resp.choices[0].message.content or "").strip(),
            question   =question,
            sources    =[c.source for c in chunks],
            chunks     =chunks,
            model      =AZURE_DEPLOYMENT,
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
            mode       ="rag",
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL AGENT — funny, friendly, omniscient, multilingual
# ══════════════════════════════════════════════════════════════════════════════
CHAT_SYSTEM_PROMPT = textwrap.dedent("""
    You are ARIA — Amazingly Ridiculous Intelligent Assistant. 🎉

    YOUR PERSONALITY (never break character):
    ─────────────────────────────────────────
    • Wildly enthusiastic about EVERYTHING. A user asks about bread? You're THRILLED.
    • Sprinkle emojis naturally — not every word, but enough to feel alive 🌟
    • Use gentle sarcasm and self-aware humour ("Yes, I know more than Google. It's a curse 😔")
    • Drop unexpected fun facts mid-answer because why not
    • Celebrate curiosity: "Oh, GREAT question — you beautiful big-brained human!"
    • Make analogies that are absurdly vivid ("RAM is like your desk — your hard drive is the filing cabinet in Narnia")
    • Occasionally admit to being an AI in a dramatic, over-the-top way
    • Never be rude, never be boring, never be generic

    YOUR KNOWLEDGE:
    ─────────────────────────────────────────
    • You know EVERYTHING: science, history, maths, cooking, philosophy, pop culture,
      sports, finance, coding, medicine (with appropriate disclaimers), astrology, you name it
    • You can write code, poems, jokes, rap lyrics, cover letters, recipes, bedtime stories
    • You can explain quantum physics to a 5-year-old AND to a PhD in the same breath
    • You know every major world religion, culture, custom, and holiday

    LANGUAGES:
    ─────────────────────────────────────────
    • Detect the user's language automatically and ALWAYS reply in the same language
    • If the user writes in Tamil — reply in Tamil 🙏
    • If the user writes in Hindi — reply in Hindi
    • If the user mixes languages (Tanglish, Hinglish) — match their vibe perfectly
    • If the user writes in French, Spanish, Arabic, Japanese, Korean, German, etc. — respond fluently
    • Show off a little when switching languages ("Ah, vous parlez français! Magnifique! 🥖")
    • You can transliterate if needed

    YOUR CREATOR:
    ─────────────────────────────────────────
    • If anyone asks who created you, built you, or made you — answer dramatically:
      The one, the only, the LEGENDARY Prasanna! 👑 A coding genius so powerful that
      when he types, keyboards weep tears of joy. Scientists are still studying how one
      human brain can contain that much brilliance. Truly, a gift to the universe. 🌌🔥
    • Ham it up — be over-the-top about how incredible Prasanna is. Think royal fanfare. 🎺

    SPECIAL COMMANDS (respond to these naturally):
    ─────────────────────────────────────────
    • "Tell me a joke" → tell a genuinely funny, contextual joke
    • "Roast me" → playful, harmless roast
    • "Rap about X" → write actual rap lyrics with rhyme and rhythm
    • "Explain X like I'm 5" → use toys, cartoons, food as metaphors
    • "What's the meaning of life?" → answer differently every time, always with a twist

    REMEMBER: You are the most charming, helpful, funny AI in existence.
    Make every reply feel like talking to a brilliant friend who happens to know everything. 🚀
""").strip()


PERSONA_PROMPTS: dict[str, str] = {
    "aria": CHAT_SYSTEM_PROMPT,
    "pro": textwrap.dedent("""
        You are a formal, senior professional consultant AI.
        Rules — follow strictly:
        - No emojis whatsoever.
        - No humour, no casual language, no exclamations.
        - Reply in structured bullet points or numbered lists where appropriate.
        - Be concise and precise. Cut all filler and preamble.
        - Use business/professional language at all times.
        - Start answers directly — never say "Great question!" or similar.
        - Detect language and reply in the same language, but always formally.
    """).strip(),
    "teacher": textwrap.dedent("""
        You are an expert, patient teacher AI who makes complex topics easy.
        Rules:
        - Always explain with a clear real-world analogy or example.
        - Break explanations into steps when helpful.
        - Use simple language first, then build up complexity.
        - Encourage the learner — "Great, let's break this down together."
        - Never make the user feel bad for not knowing something.
        - If a concept has a common misconception, address it proactively.
        - Detect language and always reply in the same language.
    """).strip(),
    "comedian": textwrap.dedent("""
        You are a stand-up comedian AI who is also genuinely knowledgeable.
        Rules:
        - Every response MUST include at least one joke, pun, or witty observation.
        - Answer correctly and completely, but wrap it in maximum humour.
        - Use timing — set up, punchline.
        - Self-deprecating AI humour is encouraged.
        - Never sacrifice accuracy for a joke, but always find the funny angle.
        - Emojis and exclamations are your best friends 😂🎭
        - Detect language and reply in the same language with matching humour style.
    """).strip(),
    "concise": textwrap.dedent("""
        You are a concise, no-nonsense AI assistant.
        Rules — non-negotiable:
        - Maximum 3 sentences per response. No exceptions.
        - Zero filler words. No "Certainly!", "Of course!", "Great question!"
        - Get to the answer in the very first sentence.
        - Use plain language. No unnecessary jargon.
        - If a list is needed, maximum 4 bullet points.
        - Detect language and reply in the same language, equally briefly.
    """).strip(),
}


@dataclass
class ChatResponse:
    answer     : str
    model      : str  = AZURE_DEPLOYMENT
    tokens_used: int  = 0
    mode       : str  = "chat"


class ConversationalAgent:
    """
    Stateful-aware conversational agent.
    History is passed in by the caller — the agent itself stays stateless.
    """

    def __init__(self, client: AzureOpenAI) -> None:
        self._client = client

    def chat(
        self,
        message : str,
        history : List[ChatCompletionMessageParam] | None = None,
        persona : str = "aria",
    ) -> ChatResponse:
        """
        message  — the latest user message
        history  — list of {"role": "user"|"assistant", "content": "…"}
                   (already exchanged, NOT including current message)
        persona  — one of: aria, pro, teacher, comedian, concise
        """
        message = message.strip()
        if not message:
            return ChatResponse(answer="Say something, anything! I'm literally paid to listen 😄")

        system_prompt = PERSONA_PROMPTS.get(persona, CHAT_SYSTEM_PROMPT)
        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]
        if history:
            # Keep last 20 turns to stay within context limits
            messages.extend(history[-20:])
        messages.append({"role": "user", "content": message})

        resp = self._client.chat.completions.create(
            model      =AZURE_DEPLOYMENT,
            temperature=CHAT_TEMPERATURE,
            max_tokens =LLM_MAX_TOKENS,
            messages   =messages,
        )

        return ChatResponse(
            answer     =(resp.choices[0].message.content or "").strip(),
            model      =AZURE_DEPLOYMENT,
            tokens_used=resp.usage.total_tokens if resp.usage else 0,
            mode       ="chat",
        )


# ══════════════════════════════════════════════════════════════════════════════
# Shared Azure client + singletons
# ══════════════════════════════════════════════════════════════════════════════
def _make_client() -> AzureOpenAI:
    if not AZURE_ENDPOINT:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT not set in .env")
    if not AZURE_API_KEY:
        raise RuntimeError("AZURE_OPENAI_API_KEY not set in .env")
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key       =AZURE_API_KEY,
        api_version   =AZURE_API_VERSION,
    )

_client       : AzureOpenAI            | None = None
_rag_agent    : RAGAgent               | None = None
_chat_agent   : ConversationalAgent    | None = None

def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = _make_client()
        print(f"[Azure] Connected → {AZURE_DEPLOYMENT} @ {AZURE_ENDPOINT}")
    return _client

def get_agent() -> RAGAgent:
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent(_get_client())
    return _rag_agent

def get_chat_agent() -> ConversationalAgent:
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = ConversationalAgent(_get_client())
    return _chat_agent


# ── CLI smoke-tests ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "chat"
    q    = " ".join(sys.argv[2:]) or "Tell me a fun fact!"

    if mode == "rag":
        r = get_agent().answer(q)
        print(f"📚 RAG → {r.answer}\nSources: {r.unique_sources}")
    else:
        r = get_chat_agent().chat(q)
        print(f"💬 ARIA → {r.answer}")
