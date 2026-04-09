"""simil CLI — find similar-sounding tracks in your local music library.

Entry point: ``simil`` (defined in pyproject.toml as ``simil.cli.main:app``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from simil.core.exceptions import SimILError

app = typer.Typer(
    name="simil",
    help="Find similar-sounding tracks in your local music library.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _abort(message: str, exit_code: int = 1) -> None:
    """Print an error and exit."""
    err_console.print(f"[red bold]Error:[/red bold] {message}")
    raise typer.Exit(exit_code)


def _get_index_dir(library: str) -> Path:
    """Return the index directory for *library*."""
    return Path.home() / ".simil" / "libraries" / library


def _require_index(index_dir: Path, library: str) -> None:
    """Abort with a helpful message if the index has not been built yet."""
    if not (index_dir / "meta.json").exists():
        _abort(
            f"No index found for library {library!r} at {index_dir}.\n"
            "  Run [bold]simil index PATH[/bold] to build it first."
        )


def _load_index_and_catalog(
    index_dir: Path,
) -> tuple[object, object]:
    """Load NumpyIndex and TrackCatalog from *index_dir*.

    Returns:
        ``(NumpyIndex, TrackCatalog)``

    Raises:
        typer.Exit: If loading fails.
    """
    from simil.catalog import TrackCatalog
    from simil.index.numpy_index import NumpyIndex

    try:
        idx = NumpyIndex.load(index_dir)
        cat = TrackCatalog.load(index_dir / "catalog.json")
    except SimILError as exc:
        _abort(str(exc))

    return idx, cat


# ── Commands ──────────────────────────────────────────────────────────────────


@app.command()
def index(
    library_path: Annotated[
        Optional[Path],
        typer.Argument(help="Path to the music library directory to scan."),
    ] = None,
    embedder: Annotated[
        str,
        typer.Option("--embedder", "-e", help="Embedder to use: mfcc, effnet-discogs, clap."),
    ] = "mfcc",
    full: Annotated[
        bool,
        typer.Option("--full", help="Force a full rebuild, ignoring the existing index."),
    ] = False,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of I/O worker threads."),
    ] = 4,
    library: Annotated[
        str,
        typer.Option("--library", "-l", help="Named library slot (default: 'default')."),
    ] = "default",
) -> None:
    """Index a music library for similarity search.

    Scans LIBRARY_PATH for audio files, embeds each track, and stores the
    vectors in a searchable index at ~/.simil/libraries/LIBRARY/.

    Re-running without --full is fast: only new or changed tracks are processed.

    \\b
    Examples:
        simil index ~/Music
        simil index ~/Music --embedder effnet-discogs
        simil index ~/Music --full --workers 8
    """
    from simil.catalog import TrackCatalog
    from simil.config import Settings
    from simil.core.exceptions import IndexEmbedderMismatch
    from simil.embedders import get_embedder
    from simil.index.numpy_index import NumpyIndex
    from simil.library.indexer import Indexer

    # ── Resolve library path ──────────────────────────────────────────────────
    env_path = os.environ.get("SIMIL_LIBRARY_PATH")
    resolved_path = library_path or (Path(env_path) if env_path else None)

    if resolved_path is None:
        _abort(
            "No library path given. Pass a path argument or set SIMIL_LIBRARY_PATH.\n"
            "  Example: simil index ~/Music"
        )

    resolved_path = Path(resolved_path).expanduser().resolve()  # type: ignore[arg-type]

    if not resolved_path.exists():
        _abort(f"Library path does not exist: {resolved_path}")
    if not resolved_path.is_dir():
        _abort(f"Library path is not a directory: {resolved_path}")

    index_dir = _get_index_dir(library)
    index_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(library_name=library, workers=workers)
    library_id = library

    # ── Print plan ────────────────────────────────────────────────────────────
    console.print(f"\n[bold]simil index[/bold]")
    console.print(f"  Library path : [cyan]{resolved_path}[/cyan]")
    console.print(f"  Index dir    : [cyan]{index_dir}[/cyan]")
    console.print(f"  Embedder     : [cyan]{embedder}[/cyan]")
    console.print(
        f"  Mode         : [cyan]{'full rebuild' if full else 'incremental'}[/cyan]\n"
    )

    # ── Load embedder ─────────────────────────────────────────────────────────
    try:
        emb = get_embedder(embedder)
    except SimILError as exc:
        _abort(str(exc))

    # ── Load or create index + catalog ────────────────────────────────────────
    meta_path = index_dir / "meta.json"
    catalog_path = index_dir / "catalog.json"

    if meta_path.exists() and not full:
        try:
            idx: NumpyIndex = NumpyIndex.load(index_dir)
        except Exception as exc:
            console.print(
                f"[yellow]Warning:[/yellow] Could not load existing index ({exc}). Starting fresh."
            )
            idx = NumpyIndex(
                embedding_dim=emb.embedding_dim,  # type: ignore[union-attr]
                embedder_name=embedder,
                library_id=library_id,
                audio_config=emb.audio_config,  # type: ignore[union-attr]
            )
    else:
        idx = NumpyIndex(
            embedding_dim=emb.embedding_dim,  # type: ignore[union-attr]
            embedder_name=embedder,
            library_id=library_id,
            audio_config=emb.audio_config,  # type: ignore[union-attr]
        )

    if catalog_path.exists() and not full:
        try:
            cat: TrackCatalog = TrackCatalog.load(catalog_path, library_id=library_id)
        except Exception as exc:
            console.print(
                f"[yellow]Warning:[/yellow] Could not load catalog ({exc}). Starting fresh."
            )
            cat = TrackCatalog(library_id=library_id)
    else:
        cat = TrackCatalog(library_id=library_id)

    indexer = Indexer(embedder=emb, index=idx, catalog=cat, settings=settings)  # type: ignore[arg-type]

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        with console.status("Indexing…", spinner="dots"):
            result = indexer.build(resolved_path, full=full)
    except IndexEmbedderMismatch as exc:
        _abort(str(exc))
    except SimILError as exc:
        _abort(str(exc))

    # ── Summary ───────────────────────────────────────────────────────────────
    tbl = Table(box=box.SIMPLE_HEAD, show_header=False)
    tbl.add_row("[green]Indexed[/green]", str(result.indexed))  # type: ignore[union-attr]
    tbl.add_row("[blue]Skipped[/blue]", str(result.skipped))  # type: ignore[union-attr]
    if result.failed:  # type: ignore[union-attr]
        tbl.add_row("[red]Failed[/red]", str(len(result.failed)))  # type: ignore[union-attr]
    tbl.add_row("Duration", f"{result.duration_seconds:.1f}s")  # type: ignore[union-attr]
    console.print(tbl)

    if result.failed:  # type: ignore[union-attr]
        console.print("[yellow]Failed files:[/yellow]")
        for p in result.failed[:10]:  # type: ignore[union-attr]
            console.print(f"  {p}")
        extra = len(result.failed) - 10  # type: ignore[union-attr]
        if extra > 0:
            console.print(f"  … and {extra} more")


@app.command()
def search(
    source: Annotated[
        str,
        typer.Argument(
            help="What to search for: a file path, URL, or free-form text description."
        ),
    ],
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of results to return."),
    ] = 10,
    min_score: Annotated[
        float,
        typer.Option("--min-score", help="Minimum cosine similarity threshold (0–1)."),
    ] = 0.0,
    library: Annotated[
        str,
        typer.Option("--library", "-l", help="Named library slot to search."),
    ] = "default",
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Output results as JSON (useful for scripting)."),
    ] = False,
) -> None:
    """Search for tracks similar to SOURCE.

    SOURCE can be:

    \\b
      • A local file path:            /path/to/song.mp3
      • A tilde path:                 ~/Music/artist/track.flac
      • A YouTube / SoundCloud URL:   https://youtu.be/...
      • A Spotify track URL:          https://open.spotify.com/track/...
      • A text description (CLAP):    "dark ambient drone"
    """
    from simil.config import Settings
    from simil.core.exceptions import ResolverError, UnsupportedURLError
    from simil.embedders import get_embedder
    from simil.search.engine import SearchEngine

    index_dir = _get_index_dir(library)
    _require_index(index_dir, library)

    idx, cat = _load_index_and_catalog(index_dir)

    # Use the embedder that matches the index to avoid dimension mismatches
    embedder_name: str = getattr(idx, "_embedder_name", None) or "mfcc"
    try:
        emb = get_embedder(embedder_name)
    except SimILError as exc:
        _abort(str(exc))

    settings = Settings(library_name=library, top_k=top_k, min_score=min_score)
    engine = SearchEngine(
        embedder=emb,  # type: ignore[arg-type]
        index=idx,  # type: ignore[arg-type]
        catalog=cat,  # type: ignore[arg-type]
        settings=settings,
    )

    try:
        with console.status("Searching…", spinner="dots"):
            results = engine.search(source, top_k=top_k, min_score=min_score)
    except (UnsupportedURLError, ResolverError, SimILError) as exc:
        _abort(str(exc))

    if not results:  # type: ignore[union-attr]
        if json_out:
            print("[]")
        else:
            console.print("[yellow]No results found.[/yellow]")
        return

    if json_out:
        import json as _json

        print(
            _json.dumps(
                [
                    {
                        "rank": r.rank,
                        "score": round(r.score, 6),
                        "raw_score": round(r.raw_score, 6),
                        "title": r.track.title,
                        "artist": r.track.artist,
                        "path": str(r.track.path),
                    }
                    for r in results  # type: ignore[union-attr]
                ],
                indent=2,
            )
        )
        return

    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    tbl.add_column("#", style="dim", width=4, justify="right")
    tbl.add_column("Score", width=7, justify="right")
    tbl.add_column("Title")
    tbl.add_column("Artist")
    tbl.add_column("Path", overflow="fold", no_wrap=False)

    for r in results:  # type: ignore[union-attr]
        tbl.add_row(
            str(r.rank),
            f"{r.score:.4f}",
            r.track.title or "[dim]—[/dim]",
            r.track.artist or "[dim]—[/dim]",
            str(r.track.path),
        )

    console.print(tbl)


@app.command()
def status(
    library: Annotated[
        str,
        typer.Option("--library", "-l", help="Named library slot to inspect."),
    ] = "default",
) -> None:
    """Show statistics for the current library index."""
    index_dir = _get_index_dir(library)
    meta_path = index_dir / "meta.json"

    if not meta_path.exists():
        console.print(
            f"[yellow]No index found[/yellow] for library [bold]{library!r}[/bold] "
            f"at {index_dir}."
        )
        console.print("Run [bold]simil index PATH[/bold] to build the index.")
        return

    meta = json.loads(meta_path.read_text())

    catalog_path = index_dir / "catalog.json"
    track_count = 0
    if catalog_path.exists():
        cat_raw = json.loads(catalog_path.read_text())
        track_count = len(cat_raw.get("tracks", []))

    disk_bytes = sum(
        p.stat().st_size for p in index_dir.iterdir() if p.is_file()
    )
    disk_mb = disk_bytes / (1024 * 1024)

    tbl = Table(box=box.SIMPLE_HEAD, show_header=False, title=f"[bold]Library: {library}[/bold]")
    tbl.add_row("Tracks", f"[green]{track_count}[/green]")
    tbl.add_row("Embedder", meta.get("embedder", "—"))
    tbl.add_row("Embedding dim", str(meta.get("embedding_dim", "—")))
    tbl.add_row("Built at", meta.get("built_at", "—"))
    tbl.add_row("Index dir", str(index_dir))
    tbl.add_row("Disk usage", f"{disk_mb:.1f} MB")
    console.print(tbl)


@app.command()
def add(
    path: Annotated[
        Path,
        typer.Argument(help="Audio file to add to the index."),
    ],
    library: Annotated[
        str,
        typer.Option("--library", "-l", help="Named library slot."),
    ] = "default",
) -> None:
    """Add a single audio file to the library index."""
    from simil.config import Settings
    from simil.core.exceptions import EmbeddingError
    from simil.core.models import Track
    from simil.embedders import get_embedder
    from simil.library.metadata import extract_metadata
    from simil.library.scanner import content_id

    path = path.expanduser().resolve()
    if not path.exists():
        _abort(f"File not found: {path}")
    if not path.is_file():
        _abort(f"Not a file: {path}")

    index_dir = _get_index_dir(library)
    _require_index(index_dir, library)

    idx, cat = _load_index_and_catalog(index_dir)

    embedder_name: str = getattr(idx, "_embedder_name", None) or "mfcc"
    try:
        emb = get_embedder(embedder_name)
    except SimILError as exc:
        _abort(str(exc))

    cid = content_id(path)
    mtime = os.path.getmtime(str(path))

    if cat.contains(cid):  # type: ignore[union-attr]
        existing = cat.get(cid)  # type: ignore[union-attr]
        if existing is not None and existing.mtime == mtime:
            console.print(f"[blue]Already indexed (unchanged):[/blue] {path.name}")
            return

    try:
        with console.status(f"Embedding {path.name}…", spinner="dots"):
            vec = emb.embed(path)  # type: ignore[union-attr]
    except EmbeddingError as exc:
        _abort(str(exc))

    meta = extract_metadata(path)
    track = Track(
        id=cid,
        path=path,
        title=meta.get("title"),  # type: ignore[arg-type]
        artist=meta.get("artist"),  # type: ignore[arg-type]
        album=meta.get("album"),  # type: ignore[arg-type]
        duration_seconds=meta.get("duration"),  # type: ignore[arg-type]
        mtime=mtime,
    )

    try:
        idx.remove(cid)  # type: ignore[union-attr]
    except Exception:
        pass

    idx.add(cid, vec)  # type: ignore[union-attr]
    cat.add(track)  # type: ignore[union-attr]

    settings = Settings(library_name=library)
    idx.save(index_dir)  # type: ignore[union-attr]
    cat.save(index_dir / "catalog.json")  # type: ignore[union-attr]

    label = track.title or path.name
    artist_str = f" by {track.artist}" if track.artist else ""
    console.print(f"[green]Added:[/green] {label}{artist_str}")


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option("--host", help="Host interface to bind."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="TCP port to listen on."),
    ] = 8000,
    library: Annotated[
        str,
        typer.Option("--library", "-l", help="Named library slot to serve."),
    ] = "default",
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Auto-reload on code changes (development only)."),
    ] = False,
) -> None:
    """Start the simil web UI server.

    Opens a local web server at http://HOST:PORT where you can drag-drop
    audio files, paste URLs, or type text descriptions to search.

    \\b
    Examples:
        simil serve
        simil serve --port 9000 --library my-library
    """
    try:
        import uvicorn
        from simil.api.app import create_app
    except ImportError as exc:
        _abort(
            f"Server dependencies not available: {exc}\n"
            "  Run: pip install simil[serve]"
        )

    from simil.config import Settings

    settings = Settings(library_name=library)

    try:
        api = create_app(settings=settings)
    except SimILError as exc:
        _abort(str(exc))

    console.print(f"\n[bold]simil serve[/bold]")
    console.print(f"  URL     : [link]http://{host}:{port}[/link]")
    console.print(f"  Library : [cyan]{library}[/cyan]")
    console.print("  Press [bold]Ctrl+C[/bold] to stop\n")

    uvicorn.run(api, host=host, port=port, reload=reload)  # type: ignore[arg-type]
