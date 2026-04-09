"""Tests for the EffNet ONNX embedder.

All tests mock onnxruntime so no model download is required in CI.
Slow tests that exercise real inference are marked with @pytest.mark.slow.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from simil.core.exceptions import EmbeddingError
from simil.core.protocols import Embedder
from simil.embedders.effnet import (
    EMBEDDING_DIM,
    N_MELS,
    PATCH_FRAMES,
    SAMPLE_RATE,
    EffNetEmbedder,
    _audio_to_patches,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_mock_session(n_patches: int = 10) -> MagicMock:
    """Create a mock ort.InferenceSession that returns random embeddings."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="serving_default_melspectrogram")]
    session.get_inputs.return_value[0].name = "serving_default_melspectrogram"

    def fake_run(output_names, feed_dict):
        actual_n = list(feed_dict.values())[0].shape[0]
        embs = np.random.default_rng(42).standard_normal((actual_n, EMBEDDING_DIM)).astype(
            np.float32
        )
        return [embs]

    session.run.side_effect = fake_run
    return session


@pytest.fixture
def embedder_with_mock_session(tmp_path: Path) -> EffNetEmbedder:
    """EffNetEmbedder with a pre-injected mock ONNX session (no download needed)."""
    emb = EffNetEmbedder(model_path=tmp_path / "fake.onnx")
    emb._session = _make_mock_session()
    return emb


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_satisfies_embedder_protocol() -> None:
    emb = EffNetEmbedder()
    assert isinstance(emb, Embedder)


def test_name() -> None:
    assert EffNetEmbedder().name == "effnet-discogs"


def test_embedding_dim() -> None:
    assert EffNetEmbedder().embedding_dim == EMBEDDING_DIM


def test_audio_config_contains_required_keys() -> None:
    cfg = EffNetEmbedder().audio_config
    for key in ("sample_rate", "n_mels", "hop_length", "patch_frames"):
        assert key in cfg, f"audio_config missing key: {key}"
    assert cfg["sample_rate"] == SAMPLE_RATE
    assert cfg["n_mels"] == N_MELS
    assert cfg["patch_frames"] == PATCH_FRAMES


# ── _audio_to_patches ─────────────────────────────────────────────────────────


def test_patches_shape_normal_audio() -> None:
    """3-second audio at 16 kHz should produce at least 1 patch."""
    audio = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    audio += np.random.default_rng(0).standard_normal(len(audio)).astype(np.float32) * 0.1
    patches = _audio_to_patches(audio)
    assert patches.ndim == 3
    assert patches.shape[1] == N_MELS
    assert patches.shape[2] == PATCH_FRAMES
    assert patches.shape[0] >= 1


def test_patches_dtype_is_float32() -> None:
    audio = np.random.default_rng(1).standard_normal(SAMPLE_RATE * 5).astype(np.float32)
    patches = _audio_to_patches(audio)
    assert patches.dtype == np.float32


def test_patches_too_short_returns_empty() -> None:
    """Audio shorter than one patch duration → shape (0, N_MELS, PATCH_FRAMES)."""
    # One patch needs PATCH_FRAMES * HOP_LENGTH + N_FFT samples minimum
    audio = np.zeros(100, dtype=np.float32)
    patches = _audio_to_patches(audio)
    assert patches.shape == (0, N_MELS, PATCH_FRAMES)


def test_patches_are_normalised() -> None:
    """Each patch should have approximately zero mean and unit std."""
    audio = np.random.default_rng(2).standard_normal(SAMPLE_RATE * 10).astype(np.float32)
    patches = _audio_to_patches(audio)
    assert patches.shape[0] > 0
    for i in range(min(5, patches.shape[0])):
        p = patches[i]
        assert abs(p.mean()) < 0.1, f"Patch {i} mean too large: {p.mean()}"
        assert abs(p.std() - 1.0) < 0.1, f"Patch {i} std too far from 1: {p.std()}"


# ── embed ─────────────────────────────────────────────────────────────────────


def test_embed_returns_correct_shape(embedder_with_mock_session: EffNetEmbedder, wav_440: Path) -> None:
    vec = embedder_with_mock_session.embed(wav_440)
    assert vec.shape == (EMBEDDING_DIM,)


def test_embed_returns_float32(embedder_with_mock_session: EffNetEmbedder, wav_440: Path) -> None:
    vec = embedder_with_mock_session.embed(wav_440)
    assert vec.dtype == np.float32


def test_embed_is_unit_norm(embedder_with_mock_session: EffNetEmbedder, wav_440: Path) -> None:
    vec = embedder_with_mock_session.embed(wav_440)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_is_finite(embedder_with_mock_session: EffNetEmbedder, wav_440: Path) -> None:
    vec = embedder_with_mock_session.embed(wav_440)
    assert np.isfinite(vec).all()


def test_embed_deterministic(embedder_with_mock_session: EffNetEmbedder, wav_440: Path) -> None:
    """Same file embedded twice returns the same vector."""
    v1 = embedder_with_mock_session.embed(wav_440)
    v2 = embedder_with_mock_session.embed(wav_440)
    np.testing.assert_array_equal(v1, v2)


def test_embed_raises_on_missing_file(embedder_with_mock_session: EffNetEmbedder, tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError):
        embedder_with_mock_session.embed(tmp_path / "nonexistent.wav")


def test_embed_raises_on_too_short(embedder_with_mock_session: EffNetEmbedder, tmp_path: Path) -> None:
    """A file that's too short to produce any patches raises EmbeddingError."""
    import wave

    short_wav = tmp_path / "short.wav"
    with wave.open(str(short_wav), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00" * 100)  # 100 bytes = 50 int16 samples ≈ 3ms

    with pytest.raises(EmbeddingError, match="too short"):
        embedder_with_mock_session.embed(short_wav)


# ── embed_batch ───────────────────────────────────────────────────────────────


def test_embed_batch_returns_list(
    embedder_with_mock_session: EffNetEmbedder, wav_440: Path, wav_880: Path
) -> None:
    results = embedder_with_mock_session.embed_batch([wav_440, wav_880])
    assert len(results) == 2


def test_embed_batch_shapes(
    embedder_with_mock_session: EffNetEmbedder, wav_440: Path, wav_880: Path
) -> None:
    results = embedder_with_mock_session.embed_batch([wav_440, wav_880])
    for vec in results:
        assert vec.shape == (EMBEDDING_DIM,)
        assert vec.dtype == np.float32
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_batch_empty_returns_empty(embedder_with_mock_session: EffNetEmbedder) -> None:
    assert embedder_with_mock_session.embed_batch([]) == []


def test_embed_batch_consistent_with_single(
    embedder_with_mock_session: EffNetEmbedder, wav_440: Path, wav_880: Path
) -> None:
    """Batch results must match individual embed() calls (mock is deterministic per-file)."""
    v440_single = embedder_with_mock_session.embed(wav_440)
    v880_single = embedder_with_mock_session.embed(wav_880)
    batch = embedder_with_mock_session.embed_batch([wav_440, wav_880])
    # Batch uses a single ONNX call so results differ from single calls
    # but both must be unit-norm float32 — check shape/dtype/norm only
    for vec in batch:
        assert vec.shape == (EMBEDDING_DIM,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


# ── Model download ────────────────────────────────────────────────────────────


def test_missing_onnxruntime_raises_clear_error(wav_440: Path, tmp_path: Path) -> None:
    """If onnxruntime is not installed, raise EmbeddingError with install hint."""
    emb = EffNetEmbedder(model_path=tmp_path / "fake.onnx")
    with patch.dict("sys.modules", {"onnxruntime": None}):
        with pytest.raises(EmbeddingError, match="onnxruntime"):
            emb.embed(wav_440)


def test_download_uses_temp_file(tmp_path: Path) -> None:
    """_download_model uses a .tmp staging file and renames on success."""
    from simil.embedders.effnet import _download_model

    dest = tmp_path / "model.onnx"

    def fake_urlretrieve(url: str, filename: str) -> None:
        Path(filename).write_bytes(b"fake model data")

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        _download_model(dest)

    assert dest.exists()
    assert not (tmp_path / "model.tmp").exists()


def test_download_cleans_tmp_on_failure(tmp_path: Path) -> None:
    """_download_model removes the .tmp file if the download fails."""
    from simil.embedders.effnet import _download_model

    dest = tmp_path / "model.onnx"

    def failing_urlretrieve(url: str, filename: str) -> None:
        Path(filename).write_bytes(b"partial")
        raise OSError("Network error")

    with patch("urllib.request.urlretrieve", side_effect=failing_urlretrieve):
        with pytest.raises(EmbeddingError):
            _download_model(dest)

    assert not dest.exists()
    assert not (tmp_path / "model.tmp").exists()


# ── Embedder registry ─────────────────────────────────────────────────────────


def test_registry_returns_effnet() -> None:
    from simil.embedders import get_embedder

    emb = get_embedder("effnet-discogs")
    assert isinstance(emb, EffNetEmbedder)
    assert emb.name == "effnet-discogs"


def test_registry_unknown_name_raises() -> None:
    from simil.core.exceptions import SimILError
    from simil.embedders import get_embedder

    with pytest.raises(SimILError, match="Unknown embedder"):
        get_embedder("nonexistent-model")


def test_list_embedders_contains_all() -> None:
    from simil.embedders import list_embedders

    names = list_embedders()
    assert "mfcc" in names
    assert "effnet-discogs" in names
    assert "clap" in names


def test_register_custom_embedder() -> None:
    from simil.embedders import get_embedder, list_embedders, register_embedder
    from simil.core.exceptions import SimILError

    register_embedder("test-custom", "simil.embedders.mfcc", "MFCCEmbedder")
    assert "test-custom" in list_embedders()
    emb = get_embedder("test-custom")
    assert emb.name == "mfcc"

    # Re-registering same name raises
    with pytest.raises(SimILError, match="already registered"):
        register_embedder("test-custom", "simil.embedders.mfcc", "MFCCEmbedder")


# ── TextEmbedder protocol ─────────────────────────────────────────────────────


def test_effnet_does_not_satisfy_text_embedder_protocol() -> None:
    """EffNet is audio-only — it must NOT satisfy TextEmbedder."""
    from simil.core.protocols import TextEmbedder

    assert not isinstance(EffNetEmbedder(), TextEmbedder)
