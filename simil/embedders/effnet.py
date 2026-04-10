"""Discogs-EffNet ONNX embedder.

Uses the MTG Essentia Discogs-EffNet (bsdynamic) model to produce 1280-dimensional
audio embeddings. This is the same model powering cosine.club's 1M+ track index.

Model details (confirmed from Essentia model JSON):
  URL:   https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bsdynamic-1.onnx
  Size:  ~18 MB
  Input: mel-spectrogram patches, shape (n_patches, 128, 96), at 16 kHz
  Output embeddings: PartitionedCall:1, shape (n_patches, 1280)
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np

from simil.audio import load_clip
from simil.core.exceptions import EmbeddingError
from simil.core.models import EmbeddingVector
from simil.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)

# ── Model constants (sourced from Essentia model JSON schema) ─────────────────

MODEL_URL = (
    "https://essentia.upf.edu/models/music-style-classification/"
    "discogs-effnet/discogs-effnet-bsdynamic-1.onnx"
)
MODEL_FILENAME = "discogs-effnet-bsdynamic-1.onnx"

# Index of the embedding output among the model's outputs.
# The model has two outputs: [0] classification logits, [1] 1280-dim embeddings.
# We reference by index rather than by name because newer ONNX Runtime versions
# reject TF-style names containing colons (e.g. "PartitionedCall:1").
EMBEDDING_OUTPUT_IDX = 1

# Audio preprocessing parameters matching Essentia's TensorflowPredictEffnetDiscogs
SAMPLE_RATE: int = 16_000
N_MELS: int = 128
N_FFT: int = 512        # 32 ms window at 16 kHz
HOP_LENGTH: int = 256   # 16 ms hop, 50% overlap
PATCH_FRAMES: int = 96  # ~1.5 s per patch (96 × 16 ms)
EMBEDDING_DIM: int = 1_280


# ── Embedder ──────────────────────────────────────────────────────────────────


class EffNetEmbedder(BaseEmbedder):
    """Discogs-EffNet audio embedder via ONNX Runtime.

    Converts audio to a 1280-dimensional embedding by:
      1. Loading audio at 16 kHz (mono, up to 3 min from the 10% mark).
      2. Computing a log-compressed mel-spectrogram (128 bins, hop=256).
      3. Windowing into non-overlapping 96-frame patches (~1.5 s each).
      4. Per-patch mean/std normalisation.
      5. Running all patches through the ONNX model in one batch call.
      6. Mean-pooling the per-patch embeddings → (1280,).
      7. L2-normalising.

    Model weights (~18 MB) are downloaded on first use and cached in
    ``~/.simil/models/`` (or ``SIMIL_MODEL_DIR`` env var).
    """

    def __init__(self, model_path: Path | None = None) -> None:
        """
        Args:
            model_path: Explicit path to the ``.onnx`` file.  If ``None``,
                the model is downloaded to ``~/.simil/models/`` on first use.
        """
        self._model_path = model_path
        self._session: object | None = None  # ort.InferenceSession, lazy-loaded
        self._embedding_output_name: str | None = None  # resolved on first use

    # ── Protocol properties ───────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "effnet-discogs"

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def audio_config(self) -> dict[str, object]:
        return {
            "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "patch_frames": PATCH_FRAMES,
            "mono": True,
        }

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed(self, audio_path: Path) -> EmbeddingVector:
        """Embed a single audio file.

        Args:
            audio_path: Path to the audio file (any format librosa can decode).

        Returns:
            L2-normalised float32 vector of shape (1280,).

        Raises:
            EmbeddingError: If audio is too short, unreadable, or inference fails.
        """
        try:
            audio = load_clip(
                audio_path,
                sample_rate=SAMPLE_RATE,
                max_duration=180.0,
                start_pct=0.10,
                mono=True,
            )
            patches = _audio_to_patches(audio)
            if patches.shape[0] == 0:
                raise EmbeddingError(
                    f"Audio too short to produce any patches: {audio_path} "
                    f"(need at least {PATCH_FRAMES * HOP_LENGTH / SAMPLE_RATE:.1f}s)"
                )

            session = self._get_session()
            input_name = session.get_inputs()[0].name
            emb_name = self._embedding_output_name
            outputs = session.run([emb_name], {input_name: patches})

            vec = outputs[0].mean(axis=0).astype(np.float32)  # (n_patches, 1280) → (1280,)
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                raise EmbeddingError(f"Zero-norm embedding for {audio_path}")
            vec = vec / norm

            self._validate_vector(vec, audio_path)
            return vec

        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"EffNet embed failed for {audio_path}: {exc}") from exc

    def embed_batch(self, audio_paths: Sequence[Path]) -> list[EmbeddingVector]:
        """Embed multiple files in a single ONNX batch call.

        All patches from all files are concatenated into one batch, which is
        more efficient than N separate ``session.run()`` calls.

        Args:
            audio_paths: Sequence of audio file paths.

        Returns:
            List of L2-normalised float32 vectors, one per input file.
        """
        if not audio_paths:
            return []

        try:
            session = self._get_session()
            input_name = session.get_inputs()[0].name

            # Load audio and compute patches for every file
            per_file_patches: list[np.ndarray] = []
            patch_counts: list[int] = []

            for path in audio_paths:
                audio = load_clip(
                    path,
                    sample_rate=SAMPLE_RATE,
                    max_duration=180.0,
                    start_pct=0.10,
                    mono=True,
                )
                patches = _audio_to_patches(audio)
                if patches.shape[0] == 0:
                    raise EmbeddingError(
                        f"Audio too short to produce any patches: {path}"
                    )
                per_file_patches.append(patches)
                patch_counts.append(patches.shape[0])

            # One ONNX call for all patches
            batch = np.concatenate(per_file_patches, axis=0)  # (total_patches, 128, 96)
            emb_name = self._embedding_output_name
            outputs = session.run([emb_name], {input_name: batch})
            all_embeddings = outputs[0]  # (total_patches, 1280)

            # Split back by file, mean-pool, L2-normalise
            results: list[EmbeddingVector] = []
            offset = 0
            for path, count in zip(audio_paths, patch_counts):
                track_embs = all_embeddings[offset : offset + count]
                vec = track_embs.mean(axis=0).astype(np.float32)
                norm = np.linalg.norm(vec)
                if norm < 1e-8:
                    raise EmbeddingError(f"Zero-norm embedding for {path}")
                vec = vec / norm
                self._validate_vector(vec, path)
                results.append(vec)
                offset += count

            return results

        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"EffNet batch embed failed: {exc}") from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_session(self) -> object:
        """Return (and lazily create) the ONNX Runtime InferenceSession."""
        if self._session is None:
            try:
                import onnxruntime as ort  # noqa: PLC0415
            except ImportError as exc:
                raise EmbeddingError(
                    "onnxruntime is not installed. Run: pip install onnxruntime"
                ) from exc

            model_path = self._model_path or _default_model_path()
            if not model_path.exists():
                logger.info("Downloading EffNet model to %s …", model_path)
                _download_model(model_path)

            logger.info("Loading EffNet ONNX model from %s", model_path)
            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            # Resolve embedding output name by index — avoids hardcoding TF-style
            # names like "PartitionedCall:1" that newer ONNX Runtime rejects.
            outputs = self._session.get_outputs()
            if len(outputs) <= EMBEDDING_OUTPUT_IDX:
                raise EmbeddingError(
                    f"EffNet model has only {len(outputs)} output(s); "
                    f"expected at least {EMBEDDING_OUTPUT_IDX + 1}. "
                    "The model file may be corrupted — delete "
                    f"{model_path} and re-run to re-download it."
                )
            self._embedding_output_name = outputs[EMBEDDING_OUTPUT_IDX].name
            logger.info("EffNet embedding output: %r", self._embedding_output_name)
        return self._session


# ── Preprocessing ─────────────────────────────────────────────────────────────


def _audio_to_patches(audio: np.ndarray) -> np.ndarray:
    """Convert a 16 kHz mono waveform to normalised mel-spectrogram patches.

    Replicates the preprocessing performed by Essentia's
    ``TensorflowPredictEffnetDiscogs`` algorithm:
      - Log-compressed mel-spectrogram (128 bins, hop=256, n_fft=512)
      - Non-overlapping patches of 96 frames (~1.5 s each)
      - Per-patch zero-mean / unit-std normalisation

    Args:
        audio: 1-D float32 waveform at SAMPLE_RATE (16 kHz).

    Returns:
        Float32 array of shape (n_patches, 128, 96).
        Returns shape (0, 128, 96) if audio is too short for even one patch.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=SAMPLE_RATE // 2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # (128, T)

    n_frames = mel_db.shape[1]
    n_patches = n_frames // PATCH_FRAMES

    if n_patches == 0:
        return np.empty((0, N_MELS, PATCH_FRAMES), dtype=np.float32)

    # Trim to exact multiple of PATCH_FRAMES, then reshape
    mel_db = mel_db[:, : n_patches * PATCH_FRAMES]          # (128, n×96)
    patches = mel_db.reshape(N_MELS, n_patches, PATCH_FRAMES)
    patches = np.ascontiguousarray(patches.transpose(1, 0, 2))  # (n, 128, 96)
    patches = patches.astype(np.float32)

    # Per-patch standardisation (zero-mean, unit-variance)
    mean = patches.mean(axis=(1, 2), keepdims=True)
    std = patches.std(axis=(1, 2), keepdims=True)
    patches = (patches - mean) / (std + 1e-6)

    return patches


# ── Model management ──────────────────────────────────────────────────────────


def _default_model_path() -> Path:
    """Return the default cached model path, creating the directory if needed."""
    cache_dir = Path(
        os.environ.get("SIMIL_MODEL_DIR", str(Path.home() / ".simil" / "models"))
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / MODEL_FILENAME


def _download_model(dest: Path) -> None:
    """Download the EffNet ONNX model to ``dest``, using a temp file for safety."""
    tmp = dest.with_suffix(".tmp")
    try:
        logger.info("Downloading EffNet model (~18 MB) from %s …", MODEL_URL)
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(dest)
        logger.info("Model saved to %s", dest)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise EmbeddingError(
            f"Failed to download EffNet model from {MODEL_URL}: {exc}\n"
            "Check your internet connection or set SIMIL_MODEL_DIR to a directory "
            "containing a pre-downloaded copy of discogs-effnet-bsdynamic-1.onnx"
        ) from exc
