# BrainBox

A personal AI knowledge base that lets you upload documents, ask questions, and get cited answers. Think NotebookLM meets ChatGPT Projects — but open source, multi-model, and you own your data.

## What makes it different

- **Web search to source** — Search the web from inside the app, preview results, and selectively import pages as knowledge sources (like NotebookLM but better)
- **Multi-model routing** — Auto-picks between Gemini Flash (free), Claude Haiku (cheap), or Claude Sonnet (smart) based on question complexity. Or choose manually per question.
- **Document comparison** — Side-by-side structured comparison of any two documents on a topic
- **Knowledge graph** — Interactive SVG-based visual map of how your documents relate (drag nodes, zoom, hover for details)
- **Custom personas** — Give BrainBox a role per knowledge base ("Business Analyst", "Devil's Advocate", etc.)
- **Structured output** — One-click formatting: bullets, tables, pros/cons, timeline, summary
- **Audio ingest** — Drop in a podcast or meeting recording, auto-transcribed and searchable
- **Citation linking** — Click any source in an answer to see the exact passages used
- **Pin answers** — Save important responses that persist across sessions
- **Full cost transparency** — See exactly what each query costs
- **Chat persistence** — Conversations saved per knowledge base, restored on reload
- **Your data stays local** — Documents stored in your Pinecone account, not in anyone else's cloud

## Quick Start

### 1. Get API Keys (5 minutes)

You need three free/cheap API keys:

| Service | What it does | Free tier? | Get key |
|---------|-------------|------------|---------|
| **Anthropic** | Generates answers (Claude) | No — ~$0.002/question with Haiku | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| **Google AI** | Embeddings + fast model + audio | Yes — generous free tier | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| **Pinecone** | Vector database (stores passages) | Yes — 100K vectors free | [app.pinecone.io](https://app.pinecone.io) |

### 2. Install & Run

```bash
git clone https://github.com/bc1224/brainbox-core.git
cd brainbox-core
```

**Windows:** Double-click `start.bat`

**Mac/Linux:** Run `./start.sh`

**Manual:** `pip install -r requirements.txt && python app.py`

On first run, a setup wizard walks you through pasting your 3 API keys. Then open **http://localhost:5000**.

### 3. Add Documents

Click **Add Sources** in the sidebar:
- **Search Web** — search for any topic and selectively import results as sources
- **Files** — drag & drop PDFs, text, markdown, HTML, JSON, CSV, code
- **URLs** — paste any webpage, Notion page, or Google Doc link
- **Folders** — point to a local folder (with optional watch mode for auto-updates)
- **Audio** — drop MP3/WAV/M4A files for automatic transcription

### 4. Ask Questions

Type a question and hit Enter. BrainBox finds relevant passages from your documents and generates a cited answer.

## Features

### Ingestion
- **Web search** — Search any topic, preview results, selectively import pages
- **File upload** — Drag & drop PDFs, text, markdown, HTML, JSON, CSV, code files
- **URL ingest** — Paste any webpage, Notion page, or Google Doc/Sheet link
- **Folder ingest** — Point to a local folder to batch-import everything
- **Audio ingest** — Drop MP3/WAV/M4A, auto-transcribed via Gemini and imported
- **Watch mode** — Auto-ingest when folder contents change
- **Scheduled refresh** — Re-ingest URLs on a timer
- **Reading list** — Bookmark URLs to ingest later

### Knowledge Management
- **Multiple knowledge bases** — Separate collections for work, research, personal, etc.
- **Tags** — Organize sources with custom tags
- **Annotations** — Add your own notes to any document
- **Pin answers** — Save important AI responses across sessions
- **Chat persistence** — Conversations saved per knowledge base, restored on reload
- **Export** — Download chat history as Markdown or JSON

### Querying
- **Multi-model routing** — Auto/Fast/Balanced/Smart model selection per question
- **Streaming answers** — Tokens appear in real-time with fade-in animation
- **Follow-up suggestions** — AI suggests 3 related questions after each answer
- **Question history** — Clock icon dropdown + arrow keys to cycle past questions
- **Synthesize** — Analyze across ALL documents at once
- **Format modes** — One-click: Bullets, Table, Pros/Cons, Timeline, Summary
- **Custom personas** — Preset roles or custom system prompts per knowledge base

### Analysis
- **Document comparison** — Pick two docs, get structured A vs B analysis on any topic
- **Knowledge graph** — Interactive SVG visualization with drag, zoom, hover tooltips
- **Citation linking** — Click a source in an answer to see the exact passages used
- **Passage viewer** — Color-coded relevance scores for every retrieved excerpt

### Settings
- **Model selection** — Haiku (~$0.002/q) or Sonnet (~$0.01/q) with cost estimates
- **Passages to retrieve** — 1-20 excerpts per question
- **Min relevance** — Filter out weak matches
- **Auto-summarize** — Generate summaries on ingest for synthesis
- **All settings persist** — Saved to localStorage, restored on reload
- **First-run tutorial** — Auto-opens for new users with step-by-step walkthrough

## Architecture

```
Browser (index.html)
    |
    | HTTP / SSE streaming
    v
Flask (app.py)
    |
    +-- Anthropic API (Claude) — answer generation
    +-- Google Gemini API — embeddings, fast queries, audio transcription
    +-- Pinecone — vector storage & similarity search
    +-- Local .summaries/ — document metadata & summaries
```

All state is stored in:
- **Pinecone** — vector embeddings (cloud, your account)
- **`.summaries/`** — document metadata (local JSON files)
- **`localStorage`** — UI settings, chat history, pinned answers (browser)

## Configuration

All settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Claude API key |
| `GOOGLE_API_KEY` | required | Gemini API key |
| `PINECONE_API_KEY` | required | Pinecone API key |
| `PINECONE_INDEX_NAME` | `brainbox` | Vector index name |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Default model |
| `CHUNK_SIZE` | `500` | Words per passage |
| `CHUNK_OVERLAP` | `50` | Overlap between passages |

## Cost

With default settings (Haiku model):
- **Per question**: ~$0.002
- **Per document ingest**: ~$0.001 (embedding) + ~$0.001 (summary)
- **Gemini Flash queries**: Free tier
- **Pinecone**: Free tier (100K vectors)

A typical session of 50 questions costs about $0.10.

## License

Apache License 2.0 — see [LICENSE](LICENSE)
