#!/usr/bin/env python3
"""
ingest.py — Ingest files into your RAG brain.

Usage:
    python ingest.py /path/to/folder
    python ingest.py /path/to/folder --namespace reshipify
    python ingest.py /path/to/single/file.pdf
    python ingest.py /path/to/folder --no-summary    # skip summarization
    python ingest.py --stats
    python ingest.py --sources                        # list all ingested docs
    python ingest.py --sources -n reshipify           # list docs in namespace
    python ingest.py --reset
    python ingest.py --reset --namespace reshipify
"""

import sys
import argparse
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

import config
import chunker
import embedder
import vectorstore
import summaries

console = Console()


def ingest_path(path: str, namespace: str = "default", summarize: bool = True):
    """Ingest a file or directory."""
    config.validate()

    p = Path(path)

    if p.is_file():
        files = [str(p)]
    elif p.is_dir():
        files = chunker.discover_files(str(p))
    else:
        console.print(f"[red]Path not found: {path}[/red]")
        return

    if not files:
        console.print("[yellow]No supported files found.[/yellow]")
        console.print(f"Supported types: {', '.join(sorted(chunker.SUPPORTED_EXTENSIONS))}")
        return

    console.print(f"\n[bold]Found {len(files)} file(s) to ingest[/bold]")
    console.print(f"Namespace: [cyan]{namespace}[/cyan]")
    if summarize:
        console.print(f"Auto-summarize: [green]on[/green]")
    console.print()

    total_chunks = 0
    file_summaries = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:

        file_task = progress.add_task("Processing files...", total=len(files))

        for filepath in files:
            fname = Path(filepath).name
            progress.update(file_task, description=f"Processing {fname}")

            # Get chunks from file
            chunks = list(chunker.process_file(
                filepath,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            ))

            if not chunks:
                progress.advance(file_task)
                continue

            # Combine full text for summarization
            full_text = " ".join(c["text"] for c in chunks) if summarize else ""

            # Embed chunks in batches
            batch_size = 20
            vectors = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [c["text"] for c in batch]

                embeddings = embedder.embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

                for chunk_item, emb in zip(batch, embeddings):
                    metadata = chunk_item["metadata"].copy()
                    metadata["text"] = chunk_item["text"][:1000]

                    vectors.append({
                        "id": chunk_item["id"],
                        "values": emb,
                        "metadata": metadata,
                    })

            # Upsert to Pinecone
            if vectors:
                vectorstore.upsert_vectors(vectors, namespace=namespace)
                total_chunks += len(vectors)

            # Generate and store summary
            if summarize and full_text:
                progress.update(file_task, description=f"Summarizing {fname}")
                try:
                    file_type = Path(filepath).suffix.lstrip(".")
                    summary = summaries.generate_summary(full_text, fname)
                    summaries.save_summary(fname, summary, file_type, len(chunks), namespace=namespace)
                    file_summaries.append((fname, summary))
                except Exception as e:
                    console.print(f"\n  [yellow]Warning: Could not summarize {fname}: {e}[/yellow]")

            progress.advance(file_task)

    console.print(f"\n[green bold]Done![/green bold] Ingested {total_chunks} chunks from {len(files)} files.\n")

    # Show summaries
    if file_summaries:
        console.print("[bold]Document Summaries:[/bold]\n")
        for fname, summary in file_summaries:
            console.print(Panel(
                summary,
                title=f"[bold cyan]{fname}[/bold cyan]",
                border_style="dim",
                width=90,
            ))
            console.print()


def show_stats():
    """Show index stats."""
    config.validate()
    stats = vectorstore.get_stats()

    console.print("\n[bold]Index Stats[/bold]\n")

    table = Table()
    table.add_column("Namespace", style="cyan")
    table.add_column("Vectors", style="green", justify="right")

    if stats.namespaces:
        for ns_name, ns_info in stats.namespaces.items():
            table.add_row(ns_name, str(ns_info.vector_count))

    table.add_row("[bold]Total[/bold]", f"[bold]{stats.total_vector_count}[/bold]")

    console.print(table)
    console.print()


def show_sources(namespace: str = "default"):
    """List all ingested documents and their summaries."""
    docs = summaries.load_summaries(namespace)

    if not docs:
        console.print(f"\n[yellow]No documents found in namespace '{namespace}'.[/yellow]\n")
        return

    console.print(f"\n[bold]Ingested Sources[/bold] — namespace: [cyan]{namespace}[/cyan]\n")

    table = Table(show_lines=True, width=90)
    table.add_column("Document", style="cyan", max_width=25)
    table.add_column("Type", style="dim", max_width=6)
    table.add_column("Chunks", justify="right", max_width=7)
    table.add_column("Summary", max_width=50)

    for fname, info in docs.items():
        table.add_row(
            fname,
            info.get("file_type", "?"),
            str(info.get("chunk_count", "?")),
            info.get("summary", "")[:150] + ("..." if len(info.get("summary", "")) > 150 else ""),
        )

    console.print(table)
    console.print()


def reset(namespace: str = "default"):
    """Delete all vectors and summaries in a namespace."""
    config.validate()

    confirm = input(f"Delete all vectors AND summaries in namespace '{namespace}'? (y/N): ")
    if confirm.lower() != "y":
        console.print("[yellow]Cancelled.[/yellow]")
        return

    vectorstore.delete_all(namespace=namespace)
    summaries.delete_summaries(namespace=namespace)
    console.print(f"[green]Cleared namespace '{namespace}'.[/green]\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest files into your RAG brain")
    parser.add_argument("path", nargs="?", help="File or directory to ingest")
    parser.add_argument("--namespace", "-n", default="default", help="Pinecone namespace (default: 'default')")
    parser.add_argument("--stats", action="store_true", help="Show index stats")
    parser.add_argument("--sources", action="store_true", help="List ingested documents")
    parser.add_argument("--reset", action="store_true", help="Delete all vectors in namespace")
    parser.add_argument("--no-summary", action="store_true", help="Skip auto-summarization")

    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.sources:
        show_sources(args.namespace)
    elif args.reset:
        reset(args.namespace)
    elif args.path:
        ingest_path(args.path, namespace=args.namespace, summarize=not args.no_summary)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
