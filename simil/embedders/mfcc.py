"""MFCC-based audio embedder."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from simil.audio import load_clip
from simil.core.exceptions import EmbeddingError
from simil.core.models import EmbeddingVector
from simil.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)

# Feature dimensions
_N_MFCC = 20      # 20 × 2 = 40 dims
_N_CHROMA = 12    # 12 × 2 = 24 dims
_N_CONTRAST = 7   # 7  × 2 = 14 dims
# Total: 78 dims

_SAMPLE_RATE = 22050
_MAX_DURATION = 180.0
_START_PCT = 0.10
_HOP_LENGTH = 512


class MFCCEmbedder(BaseEmbedder):
    """MFCC + chroma + spectral contrast embedder.

    Produces a 78-dimensional L2-normalised embedding:
    - 20 MFCCs (mean + std) = 40 dims
    - 12 chroma features (mean + std) = 24 dims
    - 7 spectral contrast bands (mean + std) = 14 dims

    All features are computed from a 180-second clip starting at 10 % of the
    file duration to avoid silence/intros.
    """

    @property
    def name(self) -> str:
        """Embedder name."""
        return "mfcc"

    @property
    def embedding_dim(self) -> int:
        """Output dimensionality (78)."""
        return 78

    @property
    def audio_config(self) -> dict[str, object]:
        """Audio loading configuration for metadata storage."""
        return {
            "sample_rate": _SAMPLE_RATE,
            "n_mels": 0,
            "hop_length": _HOP_LENGTH,
            "mono": True,
        }

    def embed(self, audio_path: Path) -> EmbeddingVector:
        """Embed an audio file using MFCC, chroma, and spectral contrast.

        Args:
            audio_path: Path to the audio file.

        Returns:
            L2-normalised float32 vector of shape ``(78,)``.

        Raises:
            EmbeddingError: If audio loading or feature extraction fails.
        """
        try:
            audio = load_clip(
                audio_path,
                sample_rate=_SAMPLE_RATE,
                max_duration=_MAX_DURATION,
                start_pct=_START_PCT,
                mono=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Failed to load audio from {audio_path!r}: {exc}") from exc

        try:
            # 20 MFCCs → 40 dims (mean + std per coefficient)
            mfcc = librosa.feature.mfcc(y=audio, sr=_SAMPLE_RATE, n_mfcc=_N_MFCC)
            mfcc_mean = mfcc.mean(axis=1)
            mfcc_std = mfcc.std(axis=1)

            # 12 chroma features → 24 dims
            chroma = librosa.feature.chroma_stft(y=audio, sr=_SAMPLE_RATE)
            chroma_mean = chroma.mean(axis=1)
            chroma_std = chroma.std(axis=1)

            # 7 spectral contrast bands → 14 dims
            contrast = librosa.feature.spectral_contrast(y=audio, sr=_SAMPLE_RATE)
            contrast_mean = contrast.mean(axis=1)
            contrast_std = contrast.std(axis=1)

        except Exception as exc:
            raise EmbeddingError(
                f"Feature extraction failed for {audio_path!r}: {exc}"
            ) from exc

        # Concatenate all features: 40 + 24 + 14 = 78
        vec = np.concatenate(
            [mfcc_mean, mfcc_std, chroma_mean, chroma_std, contrast_mean, contrast_std]
        ).astype(np.float32)

        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm > 0.0:
            vec = vec / norm

        self._validate_vector(vec, audio_path)
        logger.debug("Embedded %s → shape %s, norm=%.4f", audio_path.name, vec.shape, float(np.linalg.norm(vec)))
        return vec
