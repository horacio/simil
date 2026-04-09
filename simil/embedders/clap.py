"""LAION CLAP embedder — audio AND text queries in the same vector space.

CLAP (Contrastive Language-Audio Pretraining) maps both audio and free-form
text into a shared 512-dimensional embedding space.  This enables text-based
similarity queries: ``engine.search("dark ambient drone")`` returns tracks from
your library that match the description, using the same cosine index as audio queries.

Installation:
    pip install simil[clap]
    # Pulls in: laion-clap, torch, torchaudio

Model weights (~1 GB) are downloaded automatically on first use via laion_clap's
built-in loader.

Notes:
  - CLAP embeddings are NOT compatible with EffNet or MFCC embeddings.
    If you switch to CLAP, rebuild the index with ``simil index --full``.
  - On Apple Silicon, PyTorch uses MPS (Metal) if available; falls back to CPU.
  - Audio embed quality is broadly similar to EffNet-Discogs for genre/style
    similarity; text queries are unique to CLAP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from simil.audio import load_clip
from simil.core.exceptions import EmbeddingError
from simil.core.models import EmbeddingVector
from simil.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 48_000  # CLAP uses 48 kHz
EMBEDDING_DIM: int = 512


class CLAPEmbedder(BaseEmbedder):
    """LAION CLAP audio + text embedder.

    Implements both the ``Embedder`` and ``TextEmbedder`` protocols, enabling
    both audio-to-audio and text-to-audio similarity search.

    Requires: ``pip install simil[clap]``
    """

    def __init__(self, model_id: str = "630k-audioset-best") -> None:
        """
        Args:
            model_id: CLAP checkpoint to use.  Passed to ``laion_clap.CLAP_Module``.
                Common values: ``"630k-audioset-best"``, ``"music_audioset_epoch_15_esc_90.14"``.
        """
        self._model_id = model_id
        self._model: object | None = None

    # ── Protocol properties ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"clap-{self._model_id}"

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def audio_config(self) -> dict[str, object]:
        return {
            "sample_rate": SAMPLE_RATE,
            "n_mels": 64,  # CLAP internal
            "hop_length": 320,
            "mono": True,
        }

    # ── Audio embedding ───────────────────────────────────────────────────────

    def embed(self, audio_path: Path) -> EmbeddingVector:
        """Embed a single audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            L2-normalised float32 vector of shape (512,).
        """
        try:
            model = self._get_model()
            embeddings = model.get_audio_embedding_from_filelist(
                x=[str(audio_path)], use_tensor=False
            )
            vec = np.array(embeddings[0], dtype=np.float32)
            return self._normalise(vec, audio_path)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"CLAP embed failed for {audio_path}: {exc}") from exc

    def embed_batch(self, audio_paths: Sequence[Path]) -> list[EmbeddingVector]:
        """Embed multiple audio files in one batched CLAP call.

        Args:
            audio_paths: Sequence of audio file paths.

        Returns:
            List of L2-normalised float32 vectors.
        """
        if not audio_paths:
            return []
        try:
            model = self._get_model()
            embeddings = model.get_audio_embedding_from_filelist(
                x=[str(p) for p in audio_paths], use_tensor=False
            )
            return [
                self._normalise(np.array(e, dtype=np.float32), p)
                for e, p in zip(embeddings, audio_paths)
            ]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"CLAP batch embed failed: {exc}") from exc

    # ── Text embedding (TextEmbedder protocol) ────────────────────────────────

    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed a free-form text query into the shared audio/text space.

        Args:
            text: Natural-language description, e.g. "melancholic lo-fi jazz" or
                "heavy bass techno with distorted synths".

        Returns:
            L2-normalised float32 vector of shape (512,) in the same space as
            audio embeddings, suitable for cosine search against the index.

        Raises:
            EmbeddingError: If the model is not available or inference fails.
        """
        try:
            model = self._get_model()
            embeddings = model.get_text_embedding([text], use_tensor=False)
            vec = np.array(embeddings[0], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                raise EmbeddingError(f"Zero-norm text embedding for: {text!r}")
            return (vec / norm).astype(np.float32)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"CLAP text embed failed for {text!r}: {exc}") from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_model(self) -> object:
        if self._model is None:
            try:
                import laion_clap  # noqa: PLC0415
            except ImportError as exc:
                raise EmbeddingError(
                    "laion-clap is not installed. Run: pip install simil[clap]"
                ) from exc
            logger.info("Loading CLAP model (checkpoint: %s) …", self._model_id)
            self._model = laion_clap.CLAP_Module(enable_fusion=False)
            self._model.load_ckpt()
        return self._model

    def _normalise(self, vec: np.ndarray, path: Path) -> EmbeddingVector:
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            raise EmbeddingError(f"Zero-norm CLAP embedding for {path}")
        result = (vec / norm).astype(np.float32)
        self._validate_vector(result, path)
        return result
