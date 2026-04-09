"""Unit tests for simil.embedders.mfcc.MFCCEmbedder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simil.embedders.mfcc import MFCCEmbedder


@pytest.fixture(scope="module")
def embedder() -> MFCCEmbedder:
    """Shared MFCCEmbedder instance."""
    return MFCCEmbedder()


class TestMFCCEmbedder:
    """Tests for MFCCEmbedder."""

    def test_name(self, embedder: MFCCEmbedder) -> None:
        """Embedder name is 'mfcc'."""
        assert embedder.name == "mfcc"

    def test_embedding_dim(self, embedder: MFCCEmbedder) -> None:
        """embedding_dim is 78."""
        assert embedder.embedding_dim == 78

    def test_embed_shape(self, embedder: MFCCEmbedder, wav_440: Path) -> None:
        """embed() returns shape (78,)."""
        vec = embedder.embed(wav_440)
        assert vec.shape == (78,)

    def test_embed_dtype(self, embedder: MFCCEmbedder, wav_440: Path) -> None:
        """embed() returns float32 array."""
        vec = embedder.embed(wav_440)
        assert vec.dtype == np.float32

    def test_l2_normalised(self, embedder: MFCCEmbedder, wav_440: Path) -> None:
        """embed() returns L2-normalised vector (norm ≈ 1.0)."""
        vec = embedder.embed(wav_440)
        norm = float(np.linalg.norm(vec))
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_self_similarity(self, embedder: MFCCEmbedder, wav_440: Path) -> None:
        """Embedding the same file twice gives cosine similarity ≈ 1.0."""
        v1 = embedder.embed(wav_440)
        v2 = embedder.embed(wav_440)
        similarity = float(np.dot(v1, v2))
        assert similarity == pytest.approx(1.0, abs=1e-5)

    def test_different_tones_less_similar(
        self, embedder: MFCCEmbedder, wav_440: Path, wav_880: Path
    ) -> None:
        """Different-frequency tones have cosine similarity < 1.0."""
        v440 = embedder.embed(wav_440)
        v880 = embedder.embed(wav_880)
        similarity = float(np.dot(v440, v880))
        assert similarity < 1.0

    def test_finite_output(self, embedder: MFCCEmbedder, wav_440: Path) -> None:
        """embed() output contains only finite values."""
        vec = embedder.embed(wav_440)
        assert np.isfinite(vec).all()

    def test_audio_config_keys(self, embedder: MFCCEmbedder) -> None:
        """audio_config contains required keys."""
        cfg = embedder.audio_config
        assert "sample_rate" in cfg
        assert "hop_length" in cfg
        assert "mono" in cfg
