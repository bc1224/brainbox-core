# BrainBox Core

A personal AI knowledge base that lets you upload documents, ask questions, and get cited answers. Think NotebookLM meets ChatGPT Projects — but open source, multi-model, and you own your data.

## What makes it different

- **Multi-model routing** — Auto-picks between Gemini Flash (free), Claude Haiku (cheap), or Claude Sonnet (smart) based on question complexity. Or choose manually per question.
- **Document comparison** — Side-by-side structured comparison of any two documents on a topic
- **Knowledge graph** — Interactive visual map of how your documents relate to each other
- **Custom personas** — Give BrainBox a role per knowledge base ("Business Analyst", "Devil's Advocate", etc.)
- **Structured output** — One-click formatting: bullets, tables, pros/cons, timeline, summary
- **Audio ingest** — Drop in a podcast or meeting recording, auto-transcribed and searchable
- **Citation linking** — Click any source in an answer to see the exact passages used
- **Pin answers** — Save important responses across sessions
- **Full cost transparency** — See exactly what each query costs
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
# Clone the repo
git clone https://github.com/YOUR_USERNAME/brainbox-core.git
cd brainbox-core

# Install dependencies
pip install -r requirements.txt

# Set up your keys
cp .env.example .env
# Edit .env and paste your three API keys

# Run
python app.py
```

Open **http://localhost:5000** in your browser.

### 3. Add Documents

Click **Add Sources** in the sidebar:
- **Files** — drag & drop PDFs, text, markdown, HTML, JSON, CSV, code
- **URLs** — paste any webpage, Notion page, or Google Doc link
- **Folders** — point to a local folder (with optional watch mode for auto-updates)
- **Audio** — drop MP3/WAV/M4A files for automatic transcription

### 4. Ask Questions

Type a question and hit Enter. BrainBox finds relevant passages from your documents and generates a cited answer.

## Features

### Knowledge Management
- **Multiple knowledge bases** — Separate collections for work, research, personal, etc.
- **Tags** — Organize sources with custom tags
- **Annotations** — Add your own notes to any document
- **Reading list** — Bookmark URLs to ingest later
- **Watch mode** — Auto-ingest when folder contents change
- **Scheduled refresh** — Re-ingest URLs on a timer

### Querying
- **Streaming answers** — Tokens appear in real-time
- **Follow-up suggestions** — AI suggests related questions after each answer
- **Question history** — Arrow keys to cycle through past questions
- **Synthesize** — Analyze across ALL documents at once
- **Format modes** — Bullets, table, pros/cons, timeline, summary

### Analysis
- **Document comparison** — Pick two docs, get structured A vs B analysis
- **Knowledge graph** — Interactive visual map with drag, zoom, hover tooltips
- **Citation linking** — Click a source chip to see exactly which passages were used
- **Passage viewer** — Color-coded relevance scores for every retrieved excerpt

### Settings
- **Model selection** — Haiku (fast/cheap) or Sonnet (smart) with cost estimates
- **Passages to retrieve** — 1-20 excerpts per question
- **Min relevance** — Filter out weak matches
- **Auto-summarize** — Generate summaries on ingest for synthesis
- **Custom persona** — Preset or custom role per knowledge base

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
