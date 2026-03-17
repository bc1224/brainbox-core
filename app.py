"""
app.py — Web UI for BrainBox.

Usage:
    python app.py              # Start on http://localhost:5000
    python app.py --port 8080  # Custom port
"""

import os
import io
import csv
import uuid
import hashlib
import tempfile
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import json
import re

import requests as http_requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template, Response, stream_with_context

import config
import embedder
import vectorstore
import summaries
import chunker

import anthropic
from google import genai

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Multi-model routing
# ---------------------------------------------------------------------------
MODEL_PROFILES = {
    "fast": {"provider": "gemini", "model": "gemini-2.5-flash", "label": "Gemini Flash", "cost_in": 0.0, "cost_out": 0.0},
    "balanced": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "label": "Haiku 4.5", "cost_in": 0.80, "cost_out": 4.0},
    "smart": {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "label": "Sonnet 4", "cost_in": 3.0, "cost_out": 15.0},
}

COMPLEX_KEYWORDS = {"analyze", "compare", "contrast", "synthesize", "evaluate", "critique",
                     "explain why", "trade-off", "implications", "strategy", "recommend", "pros and cons"}


def _auto_select_model(question: str) -> str:
    """Auto-select model tier based on question complexity."""
    q = question.lower()
    # Smart: complex analysis keywords or very long questions
    if any(kw in q for kw in COMPLEX_KEYWORDS) or len(question) > 200:
        return "smart"
    # Fast: short simple questions
    if len(question.split()) <= 12:
        return "fast"
    return "balanced"

# ---------------------------------------------------------------------------
# Folder watcher state
# ---------------------------------------------------------------------------
watchers: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# URL schedule state
# ---------------------------------------------------------------------------
url_schedules: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Session-based conversation memory (in-memory, per browser session)
# ---------------------------------------------------------------------------
sessions: dict[str, list[dict]] = {}

# ---------------------------------------------------------------------------
# Usage tracking
# ---------------------------------------------------------------------------
usage = {
    "claude": {"input_tokens": 0, "output_tokens": 0, "requests": 0},
    "gemini": {"texts_embedded": 0, "requests": 0},
    "pinecone": {"queries": 0, "upserts": 0},
    "started_at": datetime.now().isoformat(),
}

# Claude pricing (per million tokens) — updates when model changes
CLAUDE_INPUT_COST = 0.80   # Haiku default
CLAUDE_OUTPUT_COST = 4.0

NO_CREDITS_MSG = "No API credits available. Please add credits at https://console.anthropic.com/settings/billing"


def _is_credit_error(exc):
    """Check if an Anthropic exception is a billing/credit error."""
    return isinstance(exc, (anthropic.BadRequestError, anthropic.AuthenticationError)) and "credit balance" in str(exc).lower()


SYSTEM_PROMPT = """You are a personal research assistant with access to the user's knowledge base. You help them understand, analyze, and synthesize information from their own documents and notes.

Rules:
- Answer based on the provided context from their documents. If the context doesn't contain the answer, say so clearly.
- Be concise and direct. No filler.
- Cite which source document(s) info comes from.
- If multiple sources conflict, note the conflict and explain the differences.
- You have access to the conversation history — use it for follow-up questions.
- When synthesizing across sources, identify themes, contradictions, and connections.
- Format in markdown for readability."""

MAX_TURNS = 20


def _get_memory(session_id: str) -> list[dict]:
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]


def _trim(messages: list[dict]):
    max_msgs = MAX_TURNS * 2
    if len(messages) > max_msgs:
        del messages[: len(messages) - max_msgs]


def _ingest_exchange(question: str, answer: str, namespace: str):
    """Embed and store a Q&A exchange into the vector store (background)."""
    try:
        text = f"Q: {question}\n\nA: {answer}"
        ts = datetime.now().isoformat()
        vec_id = hashlib.md5(f"chat:{ts}:{question[:80]}".encode()).hexdigest()[:12]

        emb = embedder.embed_texts([text], task_type="RETRIEVAL_DOCUMENT")[0]
        usage["gemini"]["texts_embedded"] += 1
        usage["gemini"]["requests"] += 1

        vectorstore.upsert_vectors([{
            "id": f"chat-{vec_id}",
            "values": emb,
            "metadata": {
                "source": "chat-history",
                "source_path": "conversation",
                "file_type": "chat",
                "chunk_index": 0,
                "total_chunks": 1,
                "ingested_at": ts,
                "text": text[:1000],
            },
        }], namespace=namespace)
        usage["pinecone"]["upserts"] += 1
    except Exception:
        pass


def _bg_ingest(question: str, answer: str, namespace: str):
    """Fire-and-forget background ingest of a Q&A exchange."""
    threading.Thread(target=_ingest_exchange, args=(question, answer, namespace), daemon=True).start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    """MD5 hash of content for duplicate detection."""
    return hashlib.md5(text.strip().encode()).hexdigest()


def _save_hash(name: str, content_hash: str, namespace: str):
    """Save content hash into summaries metadata."""
    all_sums = summaries.load_summaries(namespace)
    if name in all_sums:
        all_sums[name]["content_hash"] = content_hash
        summaries._ensure_dir()
        with open(summaries._namespace_path(namespace), "w") as f:
            json.dump(all_sums, f, indent=2)


def _check_duplicate(content_hash: str, namespace: str):
    """Check if content hash exists. Returns duplicate source name or None."""
    all_sums = summaries.load_summaries(namespace)
    for name, info in all_sums.items():
        if info.get("content_hash") == content_hash:
            return name
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


def _search_vectors(question, namespace, top_k, source_filter, min_score=0.0, query_embedding=None):
    """Shared search logic. Returns (results, source_list, context, chunk_details)."""
    if query_embedding is None:
        query_embedding = embedder.embed_query(question)
        usage["gemini"]["texts_embedded"] += 1
        usage["gemini"]["requests"] += 1

    idx = vectorstore.get_index()
    filter_dict = {"source": {"$in": source_filter}} if source_filter else None
    raw = idx.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
        filter=filter_dict,
    )
    usage["pinecone"]["queries"] += 1
    results = [{"id": m.id, "score": m.score, "metadata": m.metadata} for m in raw.matches]

    if min_score > 0:
        results = [r for r in results if r["score"] >= min_score]

    if not results:
        return [], [], "", []

    context_parts = []
    source_list = []
    chunk_details = []
    seen_sources = set()

    for match in results:
        meta = match["metadata"]
        text = meta.get("text", "")
        source = meta.get("source", "unknown")
        score = match["score"]
        context_parts.append(f"[Source: {source} | Relevance: {score:.2f}]\n{text}")

        chunk_details.append({
            "id": match["id"],
            "score": round(score, 4),
            "source": source,
            "source_path": meta.get("source_path", ""),
            "file_type": meta.get("file_type", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "total_chunks": meta.get("total_chunks", 1),
            "ingested_at": meta.get("ingested_at", ""),
            "text": text,
        })

        if source not in seen_sources:
            seen_sources.add(source)
            source_list.append({"file": source, "score": round(score, 3), "text": text[:200]})

    context = "\n\n---\n\n".join(context_parts)
    return results, source_list, context, chunk_details


@app.route("/api/query", methods=["POST"])
def api_query():
    """Ask a question (streaming via SSE)."""
    data = request.get_json()
    question = data.get("question", "").strip()
    namespace = data.get("namespace", "default")
    top_k = data.get("top_k", 5)
    session_id = data.get("session_id", "default")
    source_filter = data.get("sources", [])
    min_score = data.get("min_score", 0.0)
    model_tier = data.get("model_tier", "auto")  # auto/fast/balanced/smart
    custom_prompt = data.get("custom_prompt", "")
    format_mode = data.get("format_mode", "")  # bullets/table/pros-cons/timeline/summary

    if not question:
        return jsonify({"error": "No question provided"}), 400

    results, source_list, context, chunk_details = _search_vectors(question, namespace, top_k, source_filter, min_score)

    if not results:
        all_sums = summaries.load_summaries(namespace)
        if not all_sums:
            msg = "No documents in this knowledge base yet. Add some files or URLs using the **Add Sources** button in the sidebar, then try again."
        elif min_score > 0:
            msg = f"No documents matched above your relevance threshold ({min_score:.0%}). Try lowering the **Min relevance** in Settings, or rephrase your question."
        else:
            msg = "No strong matches found for that question. Try rephrasing, or check that you're in the right knowledge base."
        return jsonify({"answer": msg, "sources": []})

    # Resolve model tier
    if model_tier == "auto":
        model_tier = _auto_select_model(question)
    profile = MODEL_PROFILES.get(model_tier, MODEL_PROFILES["balanced"])

    # Build system prompt
    sys_prompt = SYSTEM_PROMPT
    if custom_prompt:
        sys_prompt = custom_prompt.strip() + "\n\n" + SYSTEM_PROMPT
    if format_mode:
        format_instructions = {
            "bullets": "\n\nFormat your entire response as bullet points. Use nested bullets for sub-points.",
            "table": "\n\nFormat your response using markdown tables wherever possible.",
            "pros-cons": "\n\nStructure your response as a Pros vs Cons analysis with two clear sections.",
            "timeline": "\n\nStructure your response as a chronological timeline with dates/phases.",
            "summary": "\n\nProvide a concise executive summary (3-5 sentences max), then key details.",
        }
        sys_prompt += format_instructions.get(format_mode, "")

    memory = _get_memory(session_id)
    user_content = f"Context from knowledge base:\n\n{context}\n\n---\n\nQuestion: {question}"
    memory.append({"role": "user", "content": user_content})

    def generate():
        yield f"data: {json.dumps({'type': 'sources', 'sources': source_list})}\n\n"
        yield f"data: {json.dumps({'type': 'model', 'model': profile['label'], 'tier': model_tier})}\n\n"

        full_answer = []
        try:
            if profile["provider"] == "gemini":
                # Use Gemini Flash for fast tier
                gclient = genai.Client(api_key=config.GOOGLE_API_KEY)
                # Build simple prompt for Gemini
                gemini_msgs = sys_prompt + "\n\n" + user_content
                response = gclient.models.generate_content(
                    model=profile["model"],
                    contents=gemini_msgs,
                )
                answer_text = response.text or ""
                # Stream in chunks to match SSE pattern
                for i in range(0, len(answer_text), 20):
                    chunk = answer_text[i:i+20]
                    full_answer.append(chunk)
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
                usage["gemini"]["requests"] += 1
            else:
                # Use Anthropic (Haiku or Sonnet)
                client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
                with client.messages.stream(
                    model=profile["model"],
                    max_tokens=2000,
                    system=sys_prompt,
                    messages=memory,
                ) as stream:
                    for text in stream.text_stream:
                        full_answer.append(text)
                        yield f"data: {json.dumps({'type': 'token', 'token': text})}\n\n"
                    try:
                        final_msg = stream.get_final_message()
                        if final_msg.usage:
                            usage["claude"]["input_tokens"] += final_msg.usage.input_tokens
                            usage["claude"]["output_tokens"] += final_msg.usage.output_tokens
                            usage["claude"]["requests"] += 1
                    except Exception:
                        usage["claude"]["requests"] += 1

            answer = "".join(full_answer)
            memory.append({"role": "assistant", "content": answer})
            _trim(memory)
            _bg_ingest(question, answer, namespace)

        except Exception as exc:
            if _is_credit_error(exc):
                yield f"data: {json.dumps({'type': 'token', 'token': NO_CREDITS_MSG})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'token', 'token': f'Error: {exc}'})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'chunks': chunk_details})}\n\n"

        # Generate follow-up suggestions
        try:
            suggestions = _generate_followups(question, "".join(full_answer) if full_answer else "")
            if suggestions:
                yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions})}\n\n"
        except Exception:
            pass

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


def _generate_followups(question: str, answer: str) -> list[str]:
    """Generate 3 short follow-up question suggestions based on the Q&A."""
    if not answer or len(answer) < 20:
        return []
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=150,
            system="Generate exactly 3 short follow-up questions (max 8 words each) that a user would find useful after reading this Q&A. Questions should dig deeper, explore related angles, or surface things the user hasn't considered. Return ONLY the 3 questions, one per line, no numbering or bullets.",
            messages=[{"role": "user", "content": f"Q: {question[:200]}\n\nA: {answer[:500]}"}],
        )
        lines = [l.strip() for l in msg.content[0].text.strip().split("\n") if l.strip()]
        usage["claude"]["requests"] += 1
        if msg.usage:
            usage["claude"]["input_tokens"] += msg.usage.input_tokens
            usage["claude"]["output_tokens"] += msg.usage.output_tokens
        return lines[:3]
    except Exception:
        return []


@app.route("/api/query-multi", methods=["POST"])
def api_query_multi():
    """Query across multiple namespaces (SSE streaming)."""
    data = request.get_json()
    question = data.get("question", "").strip()
    namespaces = data.get("namespaces", [])
    top_k = data.get("top_k", 5)
    session_id = data.get("session_id", "default")
    source_filter = data.get("sources", [])
    min_score = data.get("min_score", 0.0)

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not namespaces:
        return jsonify({"error": "No namespaces provided"}), 400

    # Embed once, reuse across namespaces
    query_embedding = embedder.embed_query(question)
    usage["gemini"]["texts_embedded"] += 1
    usage["gemini"]["requests"] += 1

    all_pairs = []
    for ns_name in namespaces:
        results, src_list, ctx, chunks = _search_vectors(
            question, ns_name, top_k, source_filter, min_score, query_embedding=query_embedding
        )
        for r, c in zip(results, chunks):
            r["namespace"] = ns_name
            c["namespace"] = ns_name
            all_pairs.append((r, c))

    all_pairs.sort(key=lambda x: x[0]["score"], reverse=True)
    all_pairs = all_pairs[:top_k]

    if not all_pairs:
        return jsonify({
            "answer": "No relevant documents found across the selected namespaces.",
            "sources": [],
        })

    context_parts = []
    source_list = []
    chunk_details = []
    seen_sources = set()
    for result, chunk in all_pairs:
        meta = result["metadata"]
        text = meta.get("text", "")
        source = f"[{result['namespace']}] {meta.get('source', 'unknown')}"
        score = result["score"]
        context_parts.append(f"[Source: {source} | Relevance: {score:.2f}]\n{text}")
        chunk["source_display"] = source
        chunk_details.append(chunk)
        if source not in seen_sources:
            seen_sources.add(source)
            source_list.append({"file": source, "score": round(score, 3), "text": text[:200]})

    context = "\n\n---\n\n".join(context_parts)

    memory = _get_memory(session_id)
    user_content = f"Context from knowledge base (multiple namespaces):\n\n{context}\n\n---\n\nQuestion: {question}"
    memory.append({"role": "user", "content": user_content})

    def generate():
        yield f"data: {json.dumps({'type': 'sources', 'sources': source_list})}\n\n"

        try:
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            full_answer = []
            with client.messages.stream(
                model=config.CLAUDE_MODEL, max_tokens=2000,
                system=SYSTEM_PROMPT, messages=memory,
            ) as stream:
                for text in stream.text_stream:
                    full_answer.append(text)
                    yield f"data: {json.dumps({'type': 'token', 'token': text})}\n\n"
                try:
                    final_msg = stream.get_final_message()
                    if final_msg.usage:
                        usage["claude"]["input_tokens"] += final_msg.usage.input_tokens
                        usage["claude"]["output_tokens"] += final_msg.usage.output_tokens
                        usage["claude"]["requests"] += 1
                except Exception:
                    usage["claude"]["requests"] += 1
            answer = "".join(full_answer)
            memory.append({"role": "assistant", "content": answer})
            _trim(memory)

        except Exception as exc:
            if _is_credit_error(exc):
                yield f"data: {json.dumps({'type': 'token', 'token': NO_CREDITS_MSG})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'token', 'token': f'Error: {exc}'})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'chunks': chunk_details})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/synthesize", methods=["POST"])
def api_synthesize():
    """Cross-document synthesis using summaries."""
    data = request.get_json()
    question = data.get("question", "").strip()
    namespace = data.get("namespace", "default")
    session_id = data.get("session_id", "default")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = summaries.synthesize_sources(question, namespace=namespace)
        usage["claude"]["requests"] += 1
    except Exception as exc:
        if _is_credit_error(exc):
            return jsonify({"answer": NO_CREDITS_MSG, "doc_count": 0})
        return jsonify({"answer": f"Error: {exc}", "doc_count": 0})

    if answer is None:
        return jsonify({"answer": "No document summaries found. Ingest some files first.", "doc_count": 0})

    memory = _get_memory(session_id)
    all_sums = summaries.load_summaries(namespace)
    summary_ctx = "\n".join(f"**{f}**: {info['summary']}" for f, info in all_sums.items())
    memory.append({"role": "user", "content": f"[Synthesis across all sources]\n\n{summary_ctx}\n\n---\n\nQuestion: {question}"})
    memory.append({"role": "assistant", "content": answer})
    _trim(memory)
    _bg_ingest(question, answer, namespace)

    return jsonify({"answer": answer, "doc_count": len(all_sums)})


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Compare two documents on a specific topic."""
    data = request.get_json()
    doc_a = data.get("doc_a", "").strip()
    doc_b = data.get("doc_b", "").strip()
    topic = data.get("topic", "").strip()
    namespace = data.get("namespace", "default")

    if not doc_a or not doc_b:
        return jsonify({"error": "Select two documents to compare"}), 400
    if not topic:
        return jsonify({"error": "Provide a topic or question for comparison"}), 400

    all_sums = summaries.load_summaries(namespace)
    sum_a = all_sums.get(doc_a, {}).get("summary", "No summary available")
    sum_b = all_sums.get(doc_b, {}).get("summary", "No summary available")

    # Get relevant chunks from each doc
    _, _, ctx_a, _ = _search_vectors(topic, namespace, 5, [doc_a])
    _, _, ctx_b, _ = _search_vectors(topic, namespace, 5, [doc_b])

    prompt = f"""Compare these two documents on the topic: "{topic}"

**Document A: {doc_a}**
Summary: {sum_a}
Relevant excerpts:
{ctx_a or 'No relevant passages found.'}

---

**Document B: {doc_b}**
Summary: {sum_b}
Relevant excerpts:
{ctx_b or 'No relevant passages found.'}

---

Provide a structured comparison: what each document says about the topic, where they agree, where they differ, and what each covers that the other doesn't. Use markdown formatting."""

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=config.CLAUDE_MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        usage["claude"]["requests"] += 1
        if msg.usage:
            usage["claude"]["input_tokens"] += msg.usage.input_tokens
            usage["claude"]["output_tokens"] += msg.usage.output_tokens
        return jsonify({"answer": msg.content[0].text, "doc_a": doc_a, "doc_b": doc_b, "topic": topic})
    except Exception as exc:
        if _is_credit_error(exc):
            return jsonify({"answer": NO_CREDITS_MSG})
        return jsonify({"answer": f"Error: {exc}"})


@app.route("/api/knowledge-graph", methods=["GET"])
def api_knowledge_graph():
    """Generate a knowledge graph of document relationships."""
    namespace = request.args.get("namespace", "default")
    all_sums = summaries.load_summaries(namespace)

    if len(all_sums) < 2:
        return jsonify({"nodes": [], "edges": [], "error": "Need at least 2 documents"})

    # Build nodes from summaries
    nodes = []
    for fname, info in all_sums.items():
        nodes.append({
            "id": fname,
            "label": fname.rsplit(".", 1)[0][:18] + ("..." if len(fname) > 20 else ""),
            "type": info.get("file_type", "?"),
            "chunks": info.get("chunk_count", 0),
            "summary": info.get("summary", "")[:100],
        })

    # Ask Claude to identify relationships
    summary_text = "\n".join(f"- {f}: {info.get('summary', 'No summary')[:150]}" for f, info in all_sums.items())
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=config.CLAUDE_MODEL, max_tokens=1000,
            system="You identify relationships between documents. Return ONLY a JSON array of edges. Each edge: {\"source\": \"filename\", \"target\": \"filename\", \"label\": \"short relationship description\"}. No other text.",
            messages=[{"role": "user", "content": f"Documents:\n{summary_text}\n\nIdentify all meaningful relationships between these documents. Return JSON array only."}],
        )
        usage["claude"]["requests"] += 1
        if msg.usage:
            usage["claude"]["input_tokens"] += msg.usage.input_tokens
            usage["claude"]["output_tokens"] += msg.usage.output_tokens
        # Parse edges from response
        text = msg.content[0].text.strip()
        # Extract JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            edges = json.loads(text[start:end])
        else:
            edges = []
    except Exception:
        edges = []

    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    """Upload and ingest files."""
    namespace = request.form.get("namespace", "default")
    summarize = request.form.get("summarize", "true").lower() == "true"
    force = request.form.get("force", "false").lower() == "true"
    files = request.files.getlist("files")

    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    config.validate()

    total_chunks = 0
    ingested = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in files:
            filepath = os.path.join(tmpdir, f.filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            f.save(filepath)

            chunks = list(chunker.process_file(filepath, config.CHUNK_SIZE, config.CHUNK_OVERLAP))
            if not chunks:
                continue

            full_text = " ".join(c["text"] for c in chunks)

            # Duplicate detection
            if not force:
                ch = _content_hash(full_text)
                dup = _check_duplicate(ch, namespace)
                if dup:
                    ingested.append({
                        "filename": f.filename, "chunks": 0, "summary": "",
                        "skipped": True, "duplicate_of": dup,
                    })
                    continue

            vectors = []
            batch_size = 20
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [c["text"] for c in batch]
                embeddings = embedder.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
                usage["gemini"]["texts_embedded"] += len(texts)
                usage["gemini"]["requests"] += 1
                for chunk_item, emb in zip(batch, embeddings):
                    metadata = chunk_item["metadata"].copy()
                    metadata["text"] = chunk_item["text"][:1000]
                    vectors.append({"id": chunk_item["id"], "values": emb, "metadata": metadata})

            if vectors:
                vectorstore.upsert_vectors(vectors, namespace=namespace)
                usage["pinecone"]["upserts"] += 1
                total_chunks += len(vectors)

            summary_text = ""
            if summarize and full_text:
                try:
                    file_type = Path(filepath).suffix.lstrip(".")
                    summary_text = summaries.generate_summary(full_text, f.filename)
                    summaries.save_summary(f.filename, summary_text, file_type, len(chunks), namespace=namespace)
                    usage["claude"]["requests"] += 1
                except Exception:
                    pass

            # Save content hash
            _save_hash(f.filename, _content_hash(full_text), namespace)

            ingested.append({
                "filename": f.filename,
                "chunks": len(chunks),
                "summary": summary_text,
            })

    return jsonify({"files_ingested": len(ingested), "total_chunks": total_chunks, "files": ingested})


# ---------------------------------------------------------------------------
# URL ingest helper (shared by api_ingest_url and scheduled re-ingest)
# ---------------------------------------------------------------------------

def _reingest_url(url, namespace, summarize=True):
    """Core URL ingest logic. Returns dict or None."""
    resp = http_requests.get(url, timeout=15, verify=False, headers={
        "User-Agent": "Mozilla/5.0 (BrainBox ingestion)"
    })
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)

    if not text or len(text) < 50:
        return None

    parsed = urlparse(url)
    source_name = parsed.netloc + parsed.path
    source_name = re.sub(r'[^\w\-.]', '_', source_name)[:80] + ".url"

    chunks_text = chunker.chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks_text:
        return None

    ts = datetime.now().isoformat()
    file_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    chunk_dicts = []
    for i, ct in enumerate(chunks_text):
        chunk_dicts.append({
            "id": f"{file_hash}-{i}",
            "text": ct,
            "metadata": {
                "source": source_name, "source_path": url,
                "file_type": "url", "chunk_index": i,
                "total_chunks": len(chunks_text), "ingested_at": ts,
            },
        })

    vectors = []
    batch_size = 20
    for i in range(0, len(chunk_dicts), batch_size):
        batch = chunk_dicts[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embedder.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
        usage["gemini"]["texts_embedded"] += len(texts)
        usage["gemini"]["requests"] += 1
        for chunk_item, emb in zip(batch, embeddings):
            metadata = chunk_item["metadata"].copy()
            metadata["text"] = chunk_item["text"][:1000]
            vectors.append({"id": chunk_item["id"], "values": emb, "metadata": metadata})

    if vectors:
        vectorstore.upsert_vectors(vectors, namespace=namespace)
        usage["pinecone"]["upserts"] += 1

    summary_text = ""
    if summarize:
        try:
            summary_text = summaries.generate_summary(text, source_name)
            summaries.save_summary(source_name, summary_text, "url", len(chunks_text), namespace=namespace)
            usage["claude"]["requests"] += 1
        except Exception:
            pass

    _save_hash(source_name, _content_hash(text), namespace)

    return {"source": source_name, "url": url, "chunks": len(chunks_text), "summary": summary_text}


@app.route("/api/ingest-url", methods=["POST"])
def api_ingest_url():
    """Scrape a URL and ingest its content."""
    data = request.get_json()
    url = data.get("url", "").strip()
    namespace = data.get("namespace", "default")
    force = data.get("force", False)

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    config.validate()

    try:
        result = _reingest_url(url, namespace)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch URL: {e}"}), 400

    if result is None:
        return jsonify({"error": "No meaningful content extracted from URL"}), 400

    return jsonify(result)


@app.route("/api/ingest-cloud", methods=["POST"])
def api_ingest_cloud():
    """Ingest content from Notion or Google Docs share links."""
    data = request.get_json()
    url = data.get("url", "").strip()
    namespace = data.get("namespace", "default")
    summarize = data.get("summarize", True)

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    config.validate()

    source_type = "url"
    fetch_url = url

    if "notion.so" in url or "notion.site" in url:
        source_type = "notion"
    elif "docs.google.com/document" in url:
        source_type = "gdocs"
        match = re.search(r'/document/d/([a-zA-Z0-9_-]+)', url)
        if match:
            fetch_url = f"https://docs.google.com/document/d/{match.group(1)}/export?format=txt"
        else:
            return jsonify({"error": "Could not parse Google Docs URL"}), 400
    elif "docs.google.com/spreadsheets" in url:
        source_type = "gsheets"
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', url)
        if match:
            fetch_url = f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv"
        else:
            return jsonify({"error": "Could not parse Google Sheets URL"}), 400

    try:
        resp = http_requests.get(fetch_url, timeout=20, verify=False, headers={
            "User-Agent": "Mozilla/5.0 (BrainBox ingestion)"
        }, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch: {e}"}), 400

    if source_type == "gdocs" and "format=txt" in fetch_url:
        text = resp.text.strip()
    elif source_type == "gsheets":
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)
        if rows:
            headers_row = rows[0]
            text_parts = []
            for row in rows[1:]:
                entry = ", ".join(f"{h}: {v}" for h, v in zip(headers_row, row) if v.strip())
                if entry:
                    text_parts.append(entry)
            text = "\n".join(text_parts)
        else:
            text = ""
    else:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

    if not text or len(text) < 50:
        return jsonify({"error": "No meaningful content extracted"}), 400

    parsed = urlparse(url)
    source_name = f"{source_type}_{parsed.netloc}{parsed.path}"
    source_name = re.sub(r'[^\w\-.]', '_', source_name)[:80] + f".{source_type}"

    chunks_text = chunker.chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks_text:
        return jsonify({"error": "Content too short to chunk"}), 400

    ts = datetime.now().isoformat()
    file_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    chunk_dicts = []
    for i, ct in enumerate(chunks_text):
        chunk_dicts.append({
            "id": f"{file_hash}-{i}", "text": ct,
            "metadata": {
                "source": source_name, "source_path": url,
                "file_type": source_type, "chunk_index": i,
                "total_chunks": len(chunks_text), "ingested_at": ts,
            },
        })

    vectors = []
    batch_size = 20
    for i in range(0, len(chunk_dicts), batch_size):
        batch = chunk_dicts[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embedder.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
        usage["gemini"]["texts_embedded"] += len(texts)
        usage["gemini"]["requests"] += 1
        for chunk_item, emb in zip(batch, embeddings):
            metadata = chunk_item["metadata"].copy()
            metadata["text"] = chunk_item["text"][:1000]
            vectors.append({"id": chunk_item["id"], "values": emb, "metadata": metadata})

    if vectors:
        vectorstore.upsert_vectors(vectors, namespace=namespace)
        usage["pinecone"]["upserts"] += 1

    summary_text = ""
    if summarize:
        try:
            summary_text = summaries.generate_summary(text, source_name)
            summaries.save_summary(source_name, summary_text, source_type, len(chunks_text), namespace=namespace)
            usage["claude"]["requests"] += 1
        except Exception:
            pass

    _save_hash(source_name, _content_hash(text), namespace)

    return jsonify({
        "source": source_name, "url": url,
        "source_type": source_type, "chunks": len(chunks_text),
        "summary": summary_text,
    })


@app.route("/api/source/<path:name>", methods=["DELETE"])
def api_delete_source(name):
    """Delete a single source's vectors and summary from a namespace."""
    namespace = request.args.get("namespace", "default")
    config.validate()

    idx = vectorstore.get_index()
    ids_to_delete = []
    try:
        dummy_emb = [0.0] * config.EMBEDDING_DIMENSION
        matches = idx.query(
            vector=dummy_emb, top_k=10000, include_metadata=False,
            namespace=namespace, filter={"source": {"$eq": name}},
        )
        ids_to_delete = [m.id for m in matches.matches]
        if ids_to_delete:
            for i in range(0, len(ids_to_delete), 1000):
                idx.delete(ids=ids_to_delete[i:i + 1000], namespace=namespace)
    except Exception as e:
        return jsonify({"error": f"Failed to delete vectors: {e}"}), 500

    all_sums = summaries.load_summaries(namespace)
    if name in all_sums:
        del all_sums[name]
        summaries._ensure_dir()
        with open(summaries._namespace_path(namespace), "w") as f:
            json.dump(all_sums, f, indent=2)

    return jsonify({"deleted": name, "vectors_removed": len(ids_to_delete)})


@app.route("/api/web-search", methods=["POST"])
def api_web_search():
    """Search the web and return results for the user to selectively ingest."""
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No search query provided"}), 400

    try:
        # Use Gemini to search and return results (grounded with Google Search)
        gclient = genai.Client(api_key=config.GOOGLE_API_KEY)
        response = gclient.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Search the web for: {query}\n\nReturn exactly 8 results as a JSON array. Each result must have: title, url, snippet. Return ONLY the JSON array, no other text.",
            config={
                "tools": [{"google_search": {}}],
            },
        )
        usage["gemini"]["requests"] += 1

        # Extract URLs from grounding metadata if available
        results = []
        text = response.text or ""

        # Try to parse JSON from response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end])
                for item in parsed:
                    if isinstance(item, dict) and item.get("url"):
                        results.append({
                            "title": item.get("title", item["url"]),
                            "url": item["url"],
                            "snippet": item.get("snippet", ""),
                        })
            except json.JSONDecodeError:
                pass

        # Also extract from grounding metadata
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            meta = getattr(candidate, 'grounding_metadata', None)
            if meta:
                chunks = getattr(meta, 'grounding_chunks', []) or []
                for chunk in chunks:
                    web = getattr(chunk, 'web', None)
                    if web and hasattr(web, 'uri'):
                        url = web.uri
                        title = getattr(web, 'title', url)
                        if not any(r['url'] == url for r in results):
                            results.append({"title": title, "url": url, "snippet": ""})

        if not results:
            # Fallback: just extract any URLs from the text
            urls = re.findall(r'https?://[^\s\)\]"\'<>]+', text)
            for u in urls[:10]:
                if not any(r['url'] == u for r in results):
                    results.append({"title": u.split("/")[2], "url": u, "snippet": ""})

        # Deduplicate by title and filter empty
        seen_titles = set()
        unique = []
        for r in results:
            t = r.get("title", "").strip()
            if t and t not in seen_titles and r.get("url"):
                seen_titles.add(t)
                unique.append(r)

        return jsonify({"results": unique[:10], "query": query})

    except Exception as e:
        return jsonify({"error": f"Search failed: {e}"}), 500


@app.route("/api/ingest-audio", methods=["POST"])
def api_ingest_audio():
    """Transcribe audio file using Gemini and ingest the transcript."""
    namespace = request.form.get("namespace", "default")
    summarize = request.form.get("summarize", "true").lower() == "true"
    audio_file = request.files.get("audio")

    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    config.validate()
    ext = audio_file.filename.rsplit(".", 1)[-1].lower() if "." in audio_file.filename else ""
    if ext not in ("mp3", "wav", "m4a", "ogg", "flac", "webm", "mp4"):
        return jsonify({"error": f"Unsupported audio format: .{ext}"}), 400

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Transcribe with Gemini
        gclient = genai.Client(api_key=config.GOOGLE_API_KEY)
        uploaded = gclient.files.upload(file=tmp_path)
        response = gclient.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "Transcribe this audio file completely. Return only the transcript text, no commentary.",
                uploaded,
            ],
        )
        transcript = response.text or ""
        usage["gemini"]["requests"] += 1

        if not transcript or len(transcript) < 20:
            return jsonify({"error": "Could not transcribe audio — no speech detected"}), 400

        # Ingest transcript
        source_name = audio_file.filename.rsplit(".", 1)[0] + "_transcript.txt"
        chunks_text = chunker.chunk_text(transcript, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks_text:
            return jsonify({"error": "Transcript too short to chunk"}), 400

        embeddings = []
        for i in range(0, len(chunks_text), 20):
            batch = chunks_text[i:i+20]
            embeddings.extend(embedder.embed_texts(batch, task_type="RETRIEVAL_DOCUMENT"))
            usage["gemini"]["texts_embedded"] += len(batch)
            usage["gemini"]["requests"] += 1

        vectors = []
        ts = datetime.now().isoformat()
        base_id = hashlib.md5(source_name.encode()).hexdigest()[:12]
        for i, (emb, text) in enumerate(zip(embeddings, chunks_text)):
            vectors.append({
                "id": f"{base_id}_{i}",
                "values": emb,
                "metadata": {"source": source_name, "source_path": audio_file.filename,
                             "file_type": "audio", "chunk_index": i, "total_chunks": len(chunks_text),
                             "ingested_at": ts, "text": text[:1000]},
            })

        for i in range(0, len(vectors), 100):
            vectorstore.upsert_vectors(vectors[i:i+100], namespace=namespace)
            usage["pinecone"]["upserts"] += 1

        summary = ""
        if summarize:
            summary = summaries.generate_summary(transcript[:4000], source_name)
            usage["claude"]["requests"] += 1
        summaries.save_summary(source_name, summary, "audio", len(chunks_text), namespace=namespace)

        return jsonify({"source": source_name, "chunks": len(chunks_text), "transcript_length": len(transcript), "summary": summary[:200] if summary else ""})

    except Exception as e:
        return jsonify({"error": f"Audio transcription failed: {e}"}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route("/api/ingest-folder", methods=["POST"])
def api_ingest_folder():
    """Ingest all supported files from a local folder path."""
    data = request.get_json()
    folder_path = data.get("path", "").strip()
    namespace = data.get("namespace", "default")
    summarize = data.get("summarize", True)
    force = data.get("force", False)

    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400

    p = Path(folder_path)
    if not p.exists():
        return jsonify({"error": f"Path not found: {folder_path}"}), 400

    config.validate()

    if p.is_file():
        files = [str(p)]
    elif p.is_dir():
        files = chunker.discover_files(str(p))
    else:
        return jsonify({"error": f"Invalid path: {folder_path}"}), 400

    if not files:
        return jsonify({"error": "No supported files found", "supported": sorted(chunker.SUPPORTED_EXTENSIONS)}), 400

    total_chunks = 0
    ingested = []

    for filepath in files:
        fname = Path(filepath).name
        chunks = list(chunker.process_file(filepath, config.CHUNK_SIZE, config.CHUNK_OVERLAP))
        if not chunks:
            continue

        full_text = " ".join(c["text"] for c in chunks)

        if not force:
            ch = _content_hash(full_text)
            dup = _check_duplicate(ch, namespace)
            if dup:
                ingested.append({
                    "filename": fname, "chunks": 0, "summary": "",
                    "skipped": True, "duplicate_of": dup,
                })
                continue

        vectors = []
        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = embedder.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
            usage["gemini"]["texts_embedded"] += len(texts)
            usage["gemini"]["requests"] += 1
            for chunk_item, emb in zip(batch, embeddings):
                metadata = chunk_item["metadata"].copy()
                metadata["text"] = chunk_item["text"][:1000]
                vectors.append({"id": chunk_item["id"], "values": emb, "metadata": metadata})

        if vectors:
            vectorstore.upsert_vectors(vectors, namespace=namespace)
            usage["pinecone"]["upserts"] += 1
            total_chunks += len(vectors)

        summary_text = ""
        if summarize and full_text:
            try:
                file_type = Path(filepath).suffix.lstrip(".")
                summary_text = summaries.generate_summary(full_text, fname)
                summaries.save_summary(fname, summary_text, file_type, len(chunks), namespace=namespace)
                usage["claude"]["requests"] += 1
            except Exception:
                pass

        _save_hash(fname, _content_hash(full_text), namespace)

        ingested.append({"filename": fname, "chunks": len(chunks), "summary": summary_text})

    return jsonify({"files_ingested": len(ingested), "total_chunks": total_chunks, "files": ingested})


# ---------------------------------------------------------------------------
# Folder watch mode
# ---------------------------------------------------------------------------

def _watch_folder(watcher_id: str, folder_path: str, namespace: str, stop_event: threading.Event):
    """Background thread that watches a folder for new/changed files."""
    seen: dict[str, float] = {}

    for f in chunker.discover_files(folder_path):
        seen[f] = os.path.getmtime(f)

    while not stop_event.is_set():
        stop_event.wait(10)
        if stop_event.is_set():
            break

        try:
            current_files = chunker.discover_files(folder_path)
        except Exception:
            continue

        for filepath in current_files:
            try:
                mtime = os.path.getmtime(filepath)
            except OSError:
                continue

            if filepath not in seen or mtime > seen[filepath]:
                seen[filepath] = mtime
                fname = Path(filepath).name
                try:
                    chunks = list(chunker.process_file(filepath, config.CHUNK_SIZE, config.CHUNK_OVERLAP))
                    if not chunks:
                        continue

                    full_text = " ".join(c["text"] for c in chunks)

                    vectors = []
                    batch_size = 20
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        texts = [c["text"] for c in batch]
                        embeddings = embedder.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
                        usage["gemini"]["texts_embedded"] += len(texts)
                        usage["gemini"]["requests"] += 1
                        for chunk_item, emb in zip(batch, embeddings):
                            metadata = chunk_item["metadata"].copy()
                            metadata["text"] = chunk_item["text"][:1000]
                            vectors.append({"id": chunk_item["id"], "values": emb, "metadata": metadata})

                    if vectors:
                        vectorstore.upsert_vectors(vectors, namespace=namespace)
                        usage["pinecone"]["upserts"] += 1

                    try:
                        file_type = Path(filepath).suffix.lstrip(".")
                        summary_text = summaries.generate_summary(full_text, fname)
                        summaries.save_summary(fname, summary_text, file_type, len(chunks), namespace=namespace)
                        usage["claude"]["requests"] += 1
                    except Exception:
                        pass

                    if watcher_id in watchers:
                        watchers[watcher_id].setdefault("log", []).append(
                            {"file": fname, "chunks": len(chunks), "time": datetime.now().isoformat()}
                        )
                except Exception:
                    pass

    watchers.pop(watcher_id, None)


@app.route("/api/watch", methods=["POST"])
def api_watch_start():
    """Start watching a folder for new/changed files."""
    data = request.get_json()
    folder_path = data.get("path", "").strip()
    namespace = data.get("namespace", "default")

    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400
    if not Path(folder_path).is_dir():
        return jsonify({"error": f"Not a valid directory: {folder_path}"}), 400

    for wid, w in watchers.items():
        if w["path"] == folder_path and w["namespace"] == namespace:
            return jsonify({"id": wid, "status": "already_watching"})

    watcher_id = uuid.uuid4().hex[:8]
    stop_event = threading.Event()
    t = threading.Thread(target=_watch_folder, args=(watcher_id, folder_path, namespace, stop_event), daemon=True)
    watchers[watcher_id] = {"path": folder_path, "namespace": namespace, "thread": t, "stop": stop_event, "log": []}
    t.start()

    return jsonify({"id": watcher_id, "status": "watching", "path": folder_path, "namespace": namespace})


@app.route("/api/watch", methods=["GET"])
def api_watch_list():
    result = []
    for wid, w in watchers.items():
        result.append({
            "id": wid, "path": w["path"], "namespace": w["namespace"],
            "files_ingested": len(w.get("log", [])), "recent": w.get("log", [])[-3:],
        })
    return jsonify({"watchers": result})


@app.route("/api/watch/<watcher_id>", methods=["DELETE"])
def api_watch_stop(watcher_id):
    w = watchers.get(watcher_id)
    if not w:
        return jsonify({"error": "Watcher not found"}), 404
    w["stop"].set()
    return jsonify({"stopped": watcher_id})


# ---------------------------------------------------------------------------
# URL scheduling
# ---------------------------------------------------------------------------

@app.route("/api/schedule-url", methods=["POST"])
def api_schedule_url():
    """Schedule periodic re-ingest of a URL."""
    data = request.get_json()
    url = data.get("url", "").strip()
    namespace = data.get("namespace", "default")
    interval_hours = data.get("interval_hours", 24)

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    for sid, s in url_schedules.items():
        if s["url"] == url and s["namespace"] == namespace:
            return jsonify({"id": sid, "status": "already_scheduled"})

    schedule_id = uuid.uuid4().hex[:8]
    stop_event = threading.Event()

    def run_schedule():
        while not stop_event.is_set():
            stop_event.wait(interval_hours * 3600)
            if stop_event.is_set():
                break
            try:
                _reingest_url(url, namespace)
                if schedule_id in url_schedules:
                    url_schedules[schedule_id]["last_run"] = datetime.now().isoformat()
                    url_schedules[schedule_id]["run_count"] = url_schedules[schedule_id].get("run_count", 0) + 1
            except Exception:
                pass

    t = threading.Thread(target=run_schedule, daemon=True)
    url_schedules[schedule_id] = {
        "url": url, "namespace": namespace,
        "interval_hours": interval_hours,
        "last_run": None, "run_count": 0,
        "thread": t, "stop": stop_event,
        "created_at": datetime.now().isoformat(),
    }
    t.start()

    return jsonify({"id": schedule_id, "url": url, "interval_hours": interval_hours})


@app.route("/api/schedule-url", methods=["GET"])
def api_schedule_list():
    result = []
    for sid, s in url_schedules.items():
        result.append({
            "id": sid, "url": s["url"], "namespace": s["namespace"],
            "interval_hours": s["interval_hours"], "last_run": s["last_run"],
            "run_count": s.get("run_count", 0), "created_at": s["created_at"],
        })
    return jsonify({"schedules": result})


@app.route("/api/schedule-url/<schedule_id>", methods=["DELETE"])
def api_schedule_delete(schedule_id):
    s = url_schedules.get(schedule_id)
    if not s:
        return jsonify({"error": "Schedule not found"}), 404
    s["stop"].set()
    url_schedules.pop(schedule_id, None)
    return jsonify({"stopped": schedule_id})


# ---------------------------------------------------------------------------
# Stats, sources, tags, settings
# ---------------------------------------------------------------------------

@app.route("/api/stats")
def api_stats():
    config.validate()
    stats = vectorstore.get_stats()
    ns_data = {}
    if stats.namespaces:
        for ns_name, ns_info in stats.namespaces.items():
            ns_data[ns_name] = ns_info.vector_count
    return jsonify({"namespaces": ns_data, "total": stats.total_vector_count})


@app.route("/api/sources")
def api_sources():
    namespace = request.args.get("namespace", "default")
    docs = summaries.load_summaries(namespace)
    file_list = []
    for fname, info in docs.items():
        file_list.append({
            "name": fname,
            "summary": info.get("summary", ""),
            "file_type": info.get("file_type", ""),
            "chunk_count": info.get("chunk_count", 0),
            "ingested_at": info.get("ingested_at", ""),
            "tags": info.get("tags", []),
        })
    return jsonify({"files": file_list, "namespace": namespace})


@app.route("/api/source/<path:name>/tags", methods=["PUT"])
def api_update_tags(name):
    """Update tags for a source."""
    data = request.get_json()
    namespace = data.get("namespace", "default")
    tags = data.get("tags", [])
    tags = [str(t).strip().lower()[:30] for t in tags[:10] if str(t).strip()]

    all_sums = summaries.load_summaries(namespace)
    if name not in all_sums:
        return jsonify({"error": "Source not found"}), 404

    all_sums[name]["tags"] = tags
    summaries._ensure_dir()
    with open(summaries._namespace_path(namespace), "w") as f:
        json.dump(all_sums, f, indent=2)

    return jsonify({"name": name, "tags": tags})


@app.route("/api/tags")
def api_tags():
    namespace = request.args.get("namespace", "default")
    all_sums = summaries.load_summaries(namespace)
    all_tags = set()
    for info in all_sums.values():
        all_tags.update(info.get("tags", []))
    return jsonify({"tags": sorted(all_tags)})


@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Update runtime settings."""
    global CLAUDE_INPUT_COST, CLAUDE_OUTPUT_COST
    data = request.get_json()
    if "model" in data:
        config.CLAUDE_MODEL = data["model"]
        if "haiku" in data["model"]:
            CLAUDE_INPUT_COST, CLAUDE_OUTPUT_COST = 0.80, 4.0
        else:
            CLAUDE_INPUT_COST, CLAUDE_OUTPUT_COST = 3.0, 15.0
    return jsonify({"ok": True, "model": config.CLAUDE_MODEL})


@app.route("/api/namespace/<name>", methods=["DELETE"])
def api_delete_namespace(name):
    config.validate()
    vectorstore.delete_all(namespace=name)
    summaries.delete_summaries(namespace=name)
    return jsonify({"deleted": name})


@app.route("/api/memory", methods=["DELETE"])
def api_clear_memory():
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    sessions.pop(session_id, None)
    return jsonify({"cleared": True})


# ---------------------------------------------------------------------------
# Usage tracking endpoint
# ---------------------------------------------------------------------------

@app.route("/api/usage")
def api_usage():
    in_tok = usage["claude"]["input_tokens"]
    out_tok = usage["claude"]["output_tokens"]
    claude_cost = (in_tok / 1_000_000) * CLAUDE_INPUT_COST + (out_tok / 1_000_000) * CLAUDE_OUTPUT_COST
    return jsonify({
        "claude": {
            "input_tokens": in_tok, "output_tokens": out_tok,
            "requests": usage["claude"]["requests"], "cost_usd": round(claude_cost, 6),
        },
        "gemini": {
            "texts_embedded": usage["gemini"]["texts_embedded"],
            "requests": usage["gemini"]["requests"], "cost_usd": 0,
        },
        "pinecone": {"queries": usage["pinecone"]["queries"], "upserts": usage["pinecone"]["upserts"]},
        "total_cost_usd": round(claude_cost, 6),
        "started_at": usage["started_at"],
    })


@app.route("/api/usage", methods=["DELETE"])
def api_usage_reset():
    usage["claude"] = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
    usage["gemini"] = {"texts_embedded": 0, "requests": 0}
    usage["pinecone"] = {"queries": 0, "upserts": 0}
    usage["started_at"] = datetime.now().isoformat()
    return jsonify({"reset": True})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrainBox Web UI")
    parser.add_argument("--port", "-p", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    # First-run setup check
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        print("\n  Welcome to BrainBox!")
        print("  ─────────────────────────")
        print("  No .env file found. Let's set up your API keys.\n")
        print("  You need 3 API keys (all have free tiers except Anthropic):\n")
        print("    1. Anthropic (Claude) → https://console.anthropic.com/settings/keys")
        print("    2. Google AI (Gemini) → https://aistudio.google.com/app/apikey")
        print("    3. Pinecone (vectors) → https://app.pinecone.io\n")

        example = os.path.join(os.path.dirname(__file__), ".env.example")
        if os.path.exists(example):
            import shutil
            shutil.copy(example, env_path)
            print("  Created .env from .env.example — edit it with your keys, then run again.\n")
        else:
            print("  Copy .env.example to .env, add your keys, then run again.\n")
        exit(0)

    try:
        config.validate()
    except ValueError as e:
        print(f"\n  Setup incomplete: {e}")
        print("  Edit your .env file and fill in the missing keys, then run again.\n")
        exit(1)

    print(f"\n  BrainBox -> http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=True)
