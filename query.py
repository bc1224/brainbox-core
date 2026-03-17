#!/usr/bin/env python3
"""
query.py — Research assistant for your RAG brain.

Usage:
    python query.py                              # Interactive research mode
    python query.py "what is reshipify pricing?" # Single question
    python query.py -n reshipify "pricing?"      # Query specific namespace
    python query.py -k 10 "broad question"       # More context chunks

Interactive commands:
    /ns <name>       Switch namespace
    /k <number>      Change context chunk count
    /sources         List all ingested documents
    /summary <file>  Show summary of a specific document
    /synthesize      Ask a question across ALL your documents
    /clear           Clear conversation history
    /history         Show conversation so far
    /stats           Show vector counts
    /quit            Exit
"""

import sys
import argparse
from datetime import datetime

import anthropic
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

import config
import embedder
import vectorstore
import summaries

console = Console()

SYSTEM_PROMPT = """You are a personal research assistant with access to the user's knowledge base. You help them understand, analyze, and synthesize information from their own documents and notes.

Rules:
- Answer based on the provided context from their documents. If the context doesn't contain the answer, say so clearly.
- Be concise and direct. No filler.
- Cite which source document(s) info comes from.
- If multiple sources conflict, note the conflict and explain the differences.
- You have access to the conversation history — use it for follow-up questions. If the user says "tell me more" or "what about X" or "compare that to Y", refer back to previous context.
- When synthesizing across sources, identify themes, contradictions, and connections.
- Format in markdown for readability."""


class ConversationMemory:
    """Manages conversation history for multi-turn research sessions."""

    def __init__(self, max_turns: int = 20):
        self.messages: list[dict] = []
        self.max_turns = max_turns
        self.retrieved_sources: list[dict] = []  # track what's been retrieved

    def add_user_message(self, question: str, context: str, sources: list[str]):
        """Add user question with its retrieved context."""
        # Build the full user message with context
        content = f"""Context from knowledge base:

{context}

---

Question: {question}"""

        self.messages.append({"role": "user", "content": content})
        self.retrieved_sources.append({
            "question": question,
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
        })

        self._trim()

    def add_assistant_message(self, answer: str):
        """Add assistant response."""
        self.messages.append({"role": "assistant", "content": answer})
        self._trim()

    def add_synthesis_exchange(self, question: str, context: str, answer: str):
        """Add a synthesis Q&A (uses summaries, not vector search)."""
        self.messages.append({"role": "user", "content": f"[Synthesis across all sources]\n\n{context}\n\n---\n\nQuestion: {question}"})
        self.messages.append({"role": "assistant", "content": answer})
        self._trim()

    def get_messages(self) -> list[dict]:
        """Get conversation history for API call."""
        return self.messages.copy()

    def _trim(self):
        """Keep conversation within max turns (each turn = user + assistant)."""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            # Keep the most recent messages
            self.messages = self.messages[-max_messages:]

    def clear(self):
        self.messages = []
        self.retrieved_sources = []

    def get_history_display(self) -> str:
        """Get a readable version of conversation history."""
        if not self.messages:
            return "No conversation history yet."

        lines = []
        for msg in self.messages:
            role = "You" if msg["role"] == "user" else "Brain"
            # For user messages, just show the question part
            content = msg["content"]
            if "Question:" in content:
                content = content.split("Question:")[-1].strip()
            # Truncate long responses
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"[{'cyan' if role == 'You' else 'green'}]{role}:[/] {content}")

        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        return len(self.messages) // 2


def query_with_memory(
    question: str,
    memory: ConversationMemory,
    namespace: str = "default",
    top_k: int = 5,
    show_sources: bool = True,
):
    """Ask a question with full conversation context."""

    # 1. Embed the question
    query_embedding = embedder.embed_query(question)

    # 2. Search Pinecone
    results = vectorstore.query_vectors(query_embedding, top_k=top_k, namespace=namespace)

    if not results:
        console.print("[yellow]No relevant documents found. Try ingesting some files first.[/yellow]")
        return

    # 3. Build context
    context_parts = []
    sources = set()

    for match in results:
        meta = match["metadata"]
        text = meta.get("text", "")
        source = meta.get("source", "unknown")
        score = match["score"]
        sources.add(source)
        context_parts.append(f"[Source: {source} | Relevance: {score:.2f}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    # 4. Add to memory and build messages
    memory.add_user_message(question, context, list(sources))

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    message = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=memory.get_messages(),
    )

    answer = message.content[0].text
    memory.add_assistant_message(answer)

    # 5. Display
    console.print()
    console.print(Panel(Markdown(answer), title="[bold green]Answer[/bold green]", border_style="green"))

    if show_sources:
        source_text = " · ".join(sorted(sources))
        console.print(f"\n[dim]Sources: {source_text}[/dim]")
        console.print(f"[dim]Chunks: {len(results)} | Namespace: {namespace} | Turn: {memory.turn_count}[/dim]\n")


def handle_synthesize(memory: ConversationMemory, namespace: str):
    """Handle /synthesize command — ask a question across all documents."""
    console.print("[bold]Synthesis Mode[/bold] — asking across ALL your documents\n")
    try:
        question = console.input("[bold magenta]synthesis >[/bold magenta] ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    if not question:
        return

    console.print("[dim]Analyzing all document summaries...[/dim]")
    answer = summaries.synthesize_sources(question, namespace=namespace)

    if answer is None:
        console.print("[yellow]No document summaries found. Ingest some files first.[/yellow]\n")
        return

    # Add to memory so follow-ups work
    all_summaries = summaries.load_summaries(namespace)
    summary_context = "\n".join(
        f"**{f}**: {info['summary']}" for f, info in all_summaries.items()
    )
    memory.add_synthesis_exchange(question, summary_context, answer)

    console.print()
    console.print(Panel(
        Markdown(answer),
        title="[bold magenta]Synthesis[/bold magenta]",
        border_style="magenta",
    ))

    doc_count = len(all_summaries)
    console.print(f"\n[dim]Synthesized across {doc_count} documents | Namespace: {namespace}[/dim]\n")


def handle_sources(namespace: str):
    """Show all ingested documents."""
    docs = summaries.load_summaries(namespace)

    if not docs:
        console.print(f"[yellow]No documents in namespace '{namespace}'.[/yellow]\n")
        return

    console.print(f"\n[bold]Sources in '{namespace}'[/bold]\n")
    for i, (fname, info) in enumerate(docs.items(), 1):
        console.print(f"  [cyan]{i}.[/cyan] [bold]{fname}[/bold] [dim]({info.get('file_type', '?')}, {info.get('chunk_count', '?')} chunks)[/dim]")
        console.print(f"     [dim]{info.get('summary', 'No summary')[:120]}...[/dim]")
    console.print()


def handle_summary(args: str, namespace: str):
    """Show full summary of a specific document."""
    docs = summaries.load_summaries(namespace)
    search = args.strip().lower()

    if not search:
        console.print("[yellow]Usage: /summary <filename>[/yellow]\n")
        return

    # Fuzzy match
    matches = [(f, info) for f, info in docs.items() if search in f.lower()]

    if not matches:
        console.print(f"[yellow]No document matching '{args.strip()}' found.[/yellow]")
        console.print(f"[dim]Try /sources to see all documents.[/dim]\n")
        return

    for fname, info in matches:
        console.print(Panel(
            info.get("summary", "No summary available."),
            title=f"[bold cyan]{fname}[/bold cyan]",
            subtitle=f"[dim]{info.get('file_type', '?')} · {info.get('chunk_count', '?')} chunks · ingested {info.get('ingested_at', '?')[:10]}[/dim]",
            border_style="cyan",
            width=90,
        ))
    console.print()


def interactive_mode(namespace: str, top_k: int):
    """Interactive research session with conversation memory."""
    memory = ConversationMemory(max_turns=20)

    console.print(Panel(
        "[bold]RAG Brain[/bold] — Research Mode\n"
        f"Namespace: [cyan]{namespace}[/cyan] | Top-K: {top_k}\n\n"
        "Ask questions, do follow-ups, synthesize across sources.\n\n"
        "[dim]Commands:[/dim]\n"
        "  [dim]/sources[/dim]         — list ingested docs\n"
        "  [dim]/summary <file>[/dim]  — full summary of a doc\n"
        "  [dim]/synthesize[/dim]      — ask across ALL docs\n"
        "  [dim]/ns <name>[/dim]       — switch namespace\n"
        "  [dim]/k <num>[/dim]         — change context chunks\n"
        "  [dim]/clear[/dim]           — reset conversation\n"
        "  [dim]/history[/dim]         — show conversation so far\n"
        "  [dim]/stats[/dim]           — vector counts\n"
        "  [dim]/quit[/dim]            — exit",
        border_style="blue",
    ))
    console.print()

    while True:
        try:
            question = console.input("[bold cyan]>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not question:
            continue

        # Handle commands
        if question.startswith("/"):
            parts = question.split(maxsplit=1)
            cmd = parts[0].lower()
            cmd_args = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Bye![/dim]")
                break

            elif cmd == "/ns":
                if cmd_args:
                    namespace = cmd_args.strip()
                    memory.clear()
                    console.print(f"[green]Switched to namespace: {namespace} (conversation cleared)[/green]\n")
                else:
                    console.print(f"[dim]Current namespace: {namespace}[/dim]\n")

            elif cmd == "/k":
                try:
                    top_k = int(cmd_args)
                    console.print(f"[green]Top-K set to: {top_k}[/green]\n")
                except ValueError:
                    console.print("[red]Usage: /k <number>[/red]\n")

            elif cmd == "/clear":
                memory.clear()
                console.print("[green]Conversation cleared.[/green]\n")

            elif cmd == "/history":
                console.print(Panel(
                    memory.get_history_display(),
                    title="[bold]Conversation History[/bold]",
                    border_style="dim",
                ))
                console.print()

            elif cmd == "/sources":
                handle_sources(namespace)

            elif cmd == "/summary":
                handle_summary(cmd_args, namespace)

            elif cmd == "/synthesize":
                try:
                    handle_synthesize(memory, namespace)
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]\n")

            elif cmd == "/stats":
                try:
                    stats = vectorstore.get_stats()
                    console.print(f"[dim]Total vectors: {stats.total_vector_count}[/dim]")
                    if stats.namespaces:
                        for ns, info in stats.namespaces.items():
                            console.print(f"[dim]  {ns}: {info.vector_count}[/dim]")
                    console.print()
                except Exception as e:
                    console.print(f"[red]{e}[/red]\n")

            else:
                console.print(f"[yellow]Unknown command: {cmd}[/yellow]\n")

            continue

        # Regular question — query with memory
        try:
            query_with_memory(question, memory, namespace=namespace, top_k=top_k)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")


def main():
    parser = argparse.ArgumentParser(description="Research assistant for your RAG brain")
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--namespace", "-n", default="default", help="Pinecone namespace")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of context chunks (default: 5)")
    parser.add_argument("--no-sources", action="store_true", help="Hide source info")

    args = parser.parse_args()
    config.validate()

    if args.question:
        # Single question mode — no memory needed
        memory = ConversationMemory(max_turns=1)
        query_with_memory(args.question, memory, namespace=args.namespace, top_k=args.top_k, show_sources=not args.no_sources)
    else:
        interactive_mode(args.namespace, args.top_k)


if __name__ == "__main__":
    main()
