# simil

**Music similarity search for your local library.**

Given an audio track — from a file on disk, a YouTube/Bandcamp/SoundCloud/Spotify URL, or a plain-English description — simil finds the most similar-sounding tracks in your library.

```bash
# Index once
simil index ~/Music --embedder effnet-discogs

# Search three ways
simil search ~/Music/artist/track.flac
simil search "https://youtu.be/dQw4w9WgXcQ"
simil search "dark ambient drone with field recordings"    # requires CLAP

# Or open the web UI
simil serve   →   http://127.0.0.1:8000
```

---

## Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [CLI reference](#cli-reference)
5. [Web UI](#web-ui)
6. [Embedders](#embedders)
7. [URL search](#url-search)
8. [Text search](#text-search)
9. [Spotify setup](#spotify-setup)
10. [Configuration](#configuration)
11. [Architecture](#architecture)
12. [Extending simil](#extending-simil)
13. [Development & tests](#development--tests)
14. [Troubleshooting](#troubleshooting)

---

## Features

- **File search** — drag-drop or CLI-path any audio file; finds similar tracks in your library
- **URL search** — paste a YouTube, SoundCloud, Bandcamp, Spotify, or any yt-dlp-supported link; audio is fetched, embedded, and searched on-the-fly
- **Text search** — type *"melancholic lo-fi jazz"* or *"heavy bass techno"* and get matching tracks (requires the CLAP embedder)
- **Three embedders** — MFCC (fast baseline), Discogs-EffNet (best music quality), CLAP (audio + text in one space)
- **Incremental indexing** — only re-embeds files that changed; saves a checkpoint every 100 tracks
- **Atomic writes** — index and catalog are always consistent on disk (write to `.tmp`, then rename)
- **Web UI** — drag-drop, URL paste, text query box, results grid with score bars and inline audio player, "More like this" discovery
- **CLI** — `index`, `search`, `status`, `add`, `serve`
- **192 tests**, fully mocked — no model downloads required in CI

---

## Installation

**Requirements:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/YOUR_USERNAME/simil
cd simil
uv sync
```

To install globally as a command:

```bash
uv tool install .
```

Or with pip (in a virtualenv):

```bash
pip install -e .
```

### Optional extras

```bash
# CLAP embedder (text + audio queries) — requires ~1 GB download on first use
pip install simil[clap]

# FAISS index backend (faster search at large scale, 100k+ tracks)
pip install simil[faiss]
```

### System dependencies

| Tool | Required for | Install |
|---|---|---|
| ffmpeg | URL search (yt-dlp audio extraction) | `brew install ffmpeg` / `apt install ffmpeg` |

Without ffmpeg, yt-dlp will download the raw container format (m4a, webm) and
librosa will attempt to decode it via `audioread`. Most formats will still work;
WAV/FLAC extraction requires ffmpeg.

---

## Quick start

### 1 — Index your library

```bash
# Fast start with MFCC (no model download)
simil index ~/Music

# Recommended: Discogs-EffNet (~18 MB ONNX model, downloaded automatically)
simil index ~/Music --embedder effnet-discogs

# Check what was indexed
simil status
```

The index lives in `~/.simil/libraries/default/`. Re-running is safe and fast:
only new or modified files are processed.

### 2 — Search

```bash
# By file
simil search ~/Music/Boards\ of\ Canada/Music\ Has\ the\ Right\ to\ Children/01.flac

# By URL (requires ffmpeg + internet)
simil search "https://soundcloud.com/burial/archangel"

# Show top 20 results
simil search ~/Music/track.mp3 --top-k 20

# Only high-confidence matches
simil search ~/Music/track.mp3 --min-score 0.7

# Machine-readable output
simil search ~/Music/track.mp3 --json | jq '.[0]'
```

### 3 — Web UI

```bash
simil serve
```

Open `http://127.0.0.1:8000`. From there you can:

- **Upload** a file by dragging it onto the drop zone
- **Paste a URL** in the URL tab (YouTube, SoundCloud, Bandcamp, Spotify, etc.)
- **Type a description** in the Text tab (requires CLAP)
- **Play** any result inline with the audio player
- **Click "More like this"** on any result to pivot and find tracks similar to that one

---

## CLI reference

### `simil index PATH`

Scans `PATH` for audio files (`.mp3`, `.flac`, `.wav`, `.ogg`, `.m4a`, `.opus`,
`.aac`, `.aif`, `.aiff`, `.wma`), embeds each track, and saves the index.

```
simil index PATH [OPTIONS]

Arguments:
  PATH                  Root directory to scan.
                        Falls back to SIMIL_LIBRARY_PATH env var if omitted.

Options:
  -e, --embedder TEXT   Embedder to use: mfcc, effnet-discogs, clap
                        [default: mfcc]
  --full                Force a complete rebuild, ignoring the existing index.
                        Required when switching embedders.
  -w, --workers INT     I/O worker threads for hashing + metadata.
                        [default: 4]
  -l, --library TEXT    Named library slot.  Allows multiple separate indices
                        (e.g. --library work, --library home).
                        [default: default]
```

**Examples:**

```bash
# First-time index with best-quality embedder
simil index ~/Music --embedder effnet-discogs

# Switch to CLAP (must rebuild from scratch)
simil index ~/Music --embedder clap --full

# Use a named library for a separate collection
simil index /Volumes/NAS/DJ-Sets --library djsets
```

**Incremental re-indexing:** Re-running without `--full` is fast. Each track's
`content_id` (SHA-256 of the first 160 KB) and `mtime` are compared against the
catalog. Only changed or new files are re-embedded. Deleting a file does not
automatically remove it from the index; run `simil add` or rebuild with `--full`
to purge stale entries.

---

### `simil search SOURCE`

Searches the index for tracks most similar to `SOURCE`.

```
simil search SOURCE [OPTIONS]

Arguments:
  SOURCE                What to search for.  One of:
                          • A local file path:  /path/to/song.mp3  or  ~/Music/…
                          • A URL:              https://youtu.be/…
                          • A Spotify URL:      https://open.spotify.com/track/…
                          • A text description: "dark ambient drone"  (CLAP only)

Options:
  -k, --top-k INT       Number of results to return.   [default: 10]
  --min-score FLOAT     Minimum cosine similarity (0–1 after normalisation).
                        [default: 0.0]
  -l, --library TEXT    Library to search.             [default: default]
  --json                Output results as JSON array.
```

**JSON output schema:**

```json
[
  {
    "rank": 1,
    "score": 0.9732,
    "raw_score": 0.8451,
    "title": "Roygbiv",
    "artist": "Boards of Canada",
    "path": "/home/user/Music/BoC/MHTRTC/05 Roygbiv.flac"
  }
]
```

`score` is min-max normalised within the result set (top result = 1.0).
`raw_score` is the raw cosine similarity (range: −1 to 1).

---

### `simil status`

Shows statistics for the current library index.

```
simil status [OPTIONS]

Options:
  -l, --library TEXT    Library to inspect.  [default: default]
```

Output includes: track count, embedder name, embedding dimension, index
build time, index directory, and disk usage.

---

### `simil add PATH`

Embeds a single audio file and adds it to the index without a full library scan.
Useful for adding individual tracks after the initial index is built.

```
simil add PATH [OPTIONS]

Arguments:
  PATH    Audio file to add.

Options:
  -l, --library TEXT    Target library.  [default: default]
```

If the file is already in the index and has not changed (mtime matches), it is
silently skipped.

---

### `simil serve`

Starts the web UI server.

```
simil serve [OPTIONS]

Options:
  --host TEXT           Network interface to bind.  [default: 127.0.0.1]
  -p, --port INT        TCP port.                   [default: 8000]
  -l, --library TEXT    Library to serve.           [default: default]
  --reload              Auto-reload on code changes (development only).
```

The server loads the index once on startup. Changes made by `simil index` or
`simil add` while the server is running are visible after a server restart.

---

## Web UI

Open `http://127.0.0.1:8000` after running `simil serve`.

### Search tabs

| Tab | Input | Notes |
|---|---|---|
| **Upload file** | Drag-drop or click to browse | Any format librosa can decode |
| **URL** | Paste any streaming link | See [URL search](#url-search) |
| **Text search** | Free-form description | Requires CLAP; see [Text search](#text-search) |

### Options

- **Results** (1–100) — how many similar tracks to return
- **Min score** (−1 to 1) — filter out low-confidence matches

### Result cards

Each result shows:
- Similarity score bar (0–100%)
- Title and artist (from file tags)
- Full file path
- ▶ **Play** button — streams the audio file inline via the server
- **More like this** — re-searches using that result as the new query, enabling discovery browsing

### "More like this"

Clicking "More like this" on any result fetches that track's audio from the
server and re-runs the search, pivoting the results around that track. This
lets you follow similarity chains through your library — the same mechanic
that makes cosine.club compelling.

---

## Embedders

simil ships three embedders. The index must be rebuilt with `--full` when
switching between them since their vector spaces are incompatible.

### MFCC (`mfcc`)

**Dimensions:** 78 | **Model size:** none | **Quality:** basic

Classical audio features: 20 MFCC coefficients + 12 chroma + 7 spectral
contrast, each with mean and standard deviation. Works on any hardware
instantly, no model download. Best for testing or libraries where compute
time matters more than result quality.

```bash
simil index ~/Music --embedder mfcc
```

### Discogs-EffNet (`effnet-discogs`)  ← recommended

**Dimensions:** 1280 | **Model size:** ~18 MB | **Quality:** excellent

The [MTG Essentia](https://essentia.upf.edu/) Discogs-EffNet model trained on
1.4 million Discogs releases. Produces embeddings that strongly correlate with
genre, style, mood, and instrumentation — the same model powering
[cosine.club](https://cosine.club)'s 1M+ track index.

The ONNX model is downloaded automatically on first use and cached in
`~/.simil/models/`. Runs on CPU via ONNX Runtime; no GPU required.

```bash
simil index ~/Music --embedder effnet-discogs
```

**Audio preprocessing:** 16 kHz mono → log-compressed mel-spectrogram
(128 bins, 512-point FFT, 256 hop) → non-overlapping 96-frame patches (~1.5 s
each) → per-patch mean/std normalisation → ONNX inference → mean-pool all
patches → L2-normalise.

### CLAP (`clap`)

**Dimensions:** 512 | **Model size:** ~1 GB | **Quality:** good + text

[LAION CLAP](https://github.com/LAION-AI/CLAP) maps both audio *and* free-form
text into the same vector space using contrastive learning. This enables
text-based queries: searching for "heavy bass techno" returns tracks that
actually sound like heavy bass techno.

Requires `pip install simil[clap]` (installs `laion-clap`, `torch`,
`torchaudio`). Model weights (~1 GB) are downloaded on first use.

```bash
pip install simil[clap]
simil index ~/Music --embedder clap --full
simil search "melancholic lo-fi jazz with vinyl crackle"
```

Note: CLAP audio embedding quality is broadly similar to EffNet for genre/style
similarity. The text query capability is unique to CLAP.

---

## URL search

Paste any URL into `simil search` or the web UI's URL tab. The resolver chain
handles it:

### Supported platforms

| Platform | Notes |
|---|---|
| YouTube | Requires ffmpeg for WAV extraction |
| SoundCloud | Requires ffmpeg |
| Bandcamp | Requires ffmpeg |
| Vimeo | Requires ffmpeg |
| Any yt-dlp site | 1000+ sites supported; see [yt-dlp extractors](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) |
| Spotify | 30-second preview; requires credentials (see below) |

**Only the first track of a playlist is downloaded.** For albums or playlists,
search each track individually or upload the file directly.

### How it works

1. The URL is passed to the **resolver chain** (`LocalResolver` → `SpotifyResolver` → `YtDlpResolver`).
2. yt-dlp downloads the best available audio stream to a temporary directory.
3. With ffmpeg available, audio is converted to WAV for reliable decoding.
4. The embedder processes the temporary file.
5. The temporary file is deleted.

The search result is not added to your library index — it is embedded
on-the-fly for comparison only.

### ffmpeg installation

ffmpeg is required for yt-dlp to convert downloaded audio to WAV. Without it,
yt-dlp will download in the native container format (m4a, webm, opus) and
librosa will attempt to decode it. Most formats will work; for guaranteed
compatibility, install ffmpeg:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Verify
ffmpeg -version
```

---

## Text search

Requires the CLAP embedder (see [Embedders](#embedders)).

```bash
pip install simil[clap]
simil index ~/Music --embedder clap --full
```

Then search by description:

```bash
simil search "dark ambient drone with field recordings"
simil search "upbeat 90s hip hop with jazz samples"
simil search "minimal techno 130bpm industrial"
```

Or in the web UI, use the **Text search** tab.

**How it works:** CLAP maps both audio and text into the same 512-dimensional
space using contrastive learning. The text query is encoded directly (no audio
download needed) and compared against the embedded library tracks using cosine
similarity.

**Notes:**
- Text queries only work if the library was indexed with `--embedder clap`.
  EffNet and MFCC do not support text.
- CLAP is best at broad genre/mood descriptions. Specific artist or track names
  are unlikely to work well.
- On Apple Silicon, PyTorch uses MPS (Metal) if available; falls back to CPU.

---

## Spotify setup

Spotify track previews are 30 seconds long and available for most (not all)
tracks. Some tracks have no preview — `SpotifyPreviewUnavailableError` is raised
in that case.

### 1. Create a Spotify app

1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard)
2. Click **Create App**
3. Any name and redirect URI (`http://localhost`) is fine
4. Copy the **Client ID** and **Client Secret**

### 2. Set credentials

```bash
export SIMIL_SPOTIFY_CLIENT_ID="your-client-id"
export SIMIL_SPOTIFY_CLIENT_SECRET="your-client-secret"
```

Or add to a `.env` file in the project directory:

```
SIMIL_SPOTIFY_CLIENT_ID=your-client-id
SIMIL_SPOTIFY_CLIENT_SECRET=your-client-secret
```

### 3. Search

```bash
simil search "https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh"
```

The token is cached in memory for one hour and automatically refreshed.

**Limitations:**
- Only Spotify *track* URLs are supported (not albums, playlists, or artists)
- Preview quality is 128 kbps MP3, 30 seconds — sufficient for similarity search
- Some tracks (particularly classical, certain labels, or region-locked content) have no preview

---

## Configuration

All settings can be overridden via environment variables (prefix: `SIMIL_`) or
a `.env` file in the working directory.

| Variable | Default | Description |
|---|---|---|
| `SIMIL_LIBRARY_PATH` | — | Default music library path for `simil index` |
| `SIMIL_LIBRARY_NAME` | `default` | Named library slot |
| `SIMIL_EMBEDDER` | `mfcc` | Default embedder |
| `SIMIL_WORKERS` | `4` | I/O worker threads (hashing + metadata) |
| `SIMIL_TOP_K` | `10` | Default number of search results |
| `SIMIL_MIN_SCORE` | `0.0` | Default minimum score filter |
| `SIMIL_CHECKPOINT_EVERY` | `100` | Save frequency during indexing |
| `SIMIL_SPOTIFY_CLIENT_ID` | — | Spotify API client ID |
| `SIMIL_SPOTIFY_CLIENT_SECRET` | — | Spotify API client secret |
| `SIMIL_MODEL_DIR` | `~/.simil/models/` | Directory for cached ONNX models |

**Named libraries:** Use `SIMIL_LIBRARY_NAME` (or `--library`) to maintain
multiple independent indices. Each library has its own index, catalog, and
(optionally) embedder. Common patterns:

```bash
# Home listening library with EffNet
SIMIL_LIBRARY_NAME=home simil index ~/Music --embedder effnet-discogs

# DJ archive with CLAP for text search
SIMIL_LIBRARY_NAME=djarchive simil index /Volumes/NAS/DJ-Sets --embedder clap

# Search a specific library
simil search ~/Music/track.mp3 --library home
simil serve --library djarchive
```

Index data is stored at `~/.simil/libraries/{LIBRARY_NAME}/`:

```
~/.simil/libraries/default/
  meta.json          # schema version, embedder, dimensions, build time
  vectors.npy        # float32 matrix (N × D)
  ids.json           # track_id → matrix row index
  catalog.json       # track metadata (path, title, artist, mtime, …)
  .lock              # exclusive write lock (filelock)
```

---

## Architecture

```
simil/
  core/
    protocols.py      Runtime-checkable Protocols: Embedder, Index, TextEmbedder
    models.py         Dataclasses: Track, SearchResult, IndexStats, …
    exceptions.py     Exception hierarchy rooted at SimILError

  embedders/
    base.py           BaseEmbedder ABC (embed, embed_batch, _validate_vector)
    mfcc.py           MFCCEmbedder — classical features, 78-dim
    effnet.py         EffNetEmbedder — Discogs-EffNet ONNX, 1280-dim
    clap.py           CLAPEmbedder — LAION CLAP, 512-dim, implements TextEmbedder
    __init__.py       Registry: get_embedder(), list_embedders(), register_embedder()

  resolvers/
    base.py           ResolvedAudio context manager + BaseResolver ABC
    local.py          LocalResolver — file paths and file:// URLs
    spotify.py        SpotifyResolver — 30s previews via Web API
    ytdlp.py          YtDlpResolver — yt-dlp for any http/https URL
    __init__.py       ResolverChain (local → spotify → ytdlp)

  index/
    numpy_index.py    NumpyIndex — cosine search via matrix multiply + argpartition

  catalog.py          TrackCatalog — content_id → Track metadata, JSON persistence

  library/
    scanner.py        scan_library() + content_id() (SHA-256, first 160 KB)
    metadata.py       mutagen wrapper — never raises
    indexer.py        Indexer — threaded I/O + serial inference + checkpointing

  search/
    engine.py         SearchEngine — routes Path / URL / text to correct handler

  config.py           Pydantic Settings (SIMIL_ prefix, .env support)
  audio.py            load_clip(), load_melspec() — shared audio loading

  cli/
    main.py           Typer app: index, search, status, add, serve

  api/
    app.py            FastAPI: POST /api/search, GET /api/audio/{id}, GET /api/status

  static/
    index.html        Vanilla HTML/CSS/JS web UI (no build step)
```

### Key design decisions

**`content_id` stability:** Tracks are identified by the SHA-256 of their first
160 KB of bytes. This survives renames, moves between directories, and copying
to new machines — as long as the audio data is unchanged.

**Lazy materialisation:** `NumpyIndex` accumulates new vectors in a pending list
and stacks them into the search matrix only on `search()` or `save()`. This
avoids O(N) array allocation inside the indexing loop.

**Atomic saves:** Both the index and catalog write to a `.tmp` sibling file, then
`os.replace()` atomically. A crash mid-write leaves the previous good state
intact.

**`filelock`:** A `.lock` file serialises concurrent writes from multiple processes
(e.g., two simultaneous `simil index` runs on the same library).

**Protocol + ABC split:** `Embedder`, `Index`, and `TextEmbedder` are
`@runtime_checkable Protocol`s for type-checking and `isinstance` tests.
`BaseEmbedder` is an ABC that provides the default `embed_batch()` loop and
`_validate_vector()`.

---

## Extending simil

### Add a custom embedder

```python
from simil.embedders.base import BaseEmbedder
from simil.embedders import register_embedder
from simil.core.models import EmbeddingVector
from pathlib import Path
import numpy as np

class MyEmbedder(BaseEmbedder):
    @property
    def name(self) -> str:
        return "my-embedder"

    @property
    def embedding_dim(self) -> int:
        return 256

    @property
    def audio_config(self) -> dict:
        return {"sample_rate": 22050, "mono": True}

    def embed(self, audio_path: Path) -> EmbeddingVector:
        # ... your inference code ...
        vec = np.zeros(256, dtype=np.float32)
        return vec / np.linalg.norm(vec)

# Register globally
register_embedder("my-embedder", "mypackage.myembedder", "MyEmbedder")

# Then use it
simil index ~/Music --embedder my-embedder
```

### Add a custom resolver

```python
from simil.resolvers.base import BaseResolver, ResolvedAudio
from simil.resolvers import ResolverChain
from simil.core.exceptions import ResolverError
from pathlib import Path

class MyResolver(BaseResolver):
    @property
    def name(self) -> str:
        return "myservice"

    def can_handle(self, source: str) -> bool:
        return source.startswith("myservice://")

    def resolve(self, source: str) -> ResolvedAudio:
        # ... download to temp file ...
        return ResolvedAudio(path=Path("/tmp/audio.wav"), origin=source, _tmp_dir=...)

# Use in a custom chain
chain = ResolverChain(resolvers=[MyResolver(), *ResolverChain.default_resolvers()])
```

### Text-capable embedder

Any embedder that also satisfies the `TextEmbedder` protocol (i.e. has an
`embed_text(text: str) -> EmbeddingVector` method) will automatically enable
text search in `simil search` and the web UI.

---

## Development & tests

```bash
uv sync --extra dev

# Run all 192 tests (~2 seconds)
uv run pytest

# With coverage
uv run pytest --cov=simil --cov-report=term-missing

# Run only a specific module
uv run pytest tests/unit/test_effnet_embedder.py -v
```

**Test philosophy:** All tests are fully mocked — no model downloads, no network
access, no files written outside `tmp_path`. The ONNX session in the EffNet tests
is replaced by a mock that returns random vectors. URL resolvers are mocked with
`patch`. The CLI and API tests use `typer.testing.CliRunner` and
`fastapi.testclient.TestClient` respectively.

**Slow tests** (real inference, real model downloads) are marked `@pytest.mark.slow`
and skipped by default:

```bash
uv run pytest -m slow   # run slow tests (requires internet + model download)
```

---

## Troubleshooting

### "No index found for library 'default'"

You haven't indexed a library yet. Run:

```bash
simil index ~/Music
```

### "yt-dlp download failed" / "ffmpeg not found"

Install ffmpeg:

```bash
brew install ffmpeg         # macOS
sudo apt install ffmpeg     # Debian/Ubuntu
```

### "Spotify credentials are required"

Set `SIMIL_SPOTIFY_CLIENT_ID` and `SIMIL_SPOTIFY_CLIENT_SECRET`. See
[Spotify setup](#spotify-setup).

### "SpotifyPreviewUnavailableError"

The track has no 30-second preview. This is common for classical music,
tracks by certain major labels, or region-locked content. Try a YouTube
or SoundCloud link instead.

### "Index was built with embedder 'mfcc' but current embedder is 'effnet-discogs'"

You switched embedders without rebuilding. Run:

```bash
simil index ~/Music --embedder effnet-discogs --full
```

### "EmbeddingError: Audio too short"

The file is too short to produce even one mel-spectrogram patch (~0.4 seconds
for EffNet). Skip it or find a longer version.

### "laion-clap is not installed"

```bash
pip install simil[clap]
```

### "onnxruntime is not installed"

```bash
pip install onnxruntime
```

This is included in the default install. If missing, the package install may
have been partial.

### Results look wrong / very generic

Switch to a better embedder:

```bash
simil index ~/Music --embedder effnet-discogs --full
```

MFCC captures basic spectral character but is not music-aware. EffNet is
trained on Discogs genre labels and captures genre, subgenre, instrumentation,
and production style much more accurately.

### Web UI shows "Could not connect to server"

The server is not running. Start it with `simil serve`, then refresh the page.

### Performance

- **EffNet indexing:** ~2–5 tracks/second on a modern CPU (librosa + ONNX inference)
- **Search speed:** sub-millisecond for up to ~100k tracks (pure NumPy matrix multiply)
- **URL search latency:** depends on yt-dlp download speed (typically 3–15 seconds)
- **Memory:** ~5 MB per 1000 tracks at 1280 dimensions (float32)
