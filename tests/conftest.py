"""Shared test fixtures for simil."""

from __future__ import annotations

import hashlib
import struct
import wave
from pathlib import Path

import numpy as np
import pytest


def _write_wav(path: Path, freq: float, duration: float = 3.0, sr: int = 22050) -> None:
    """Write a synthetic sine-wave WAV file."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


@pytest.fixture(scope="session")
def wav_440(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3-second 440 Hz sine wave WAV."""
    p = tmp_path_factory.mktemp("audio") / "sine_440.wav"
    _write_wav(p, 440.0)
    return p


@pytest.fixture(scope="session")
def wav_880(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3-second 880 Hz sine wave WAV."""
    p = tmp_path_factory.mktemp("audio") / "sine_880.wav"
    _write_wav(p, 880.0)
    return p


@pytest.fixture(scope="session")
def wav_220(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3-second 220 Hz sine wave WAV."""
    p = tmp_path_factory.mktemp("audio") / "sine_220.wav"
    _write_wav(p, 220.0)
    return p


@pytest.fixture
def mock_embedder() -> object:
    """A deterministic mock embedder backed by SHA-256 hashing."""

    class MockEmbedder:
        name = "mock"
        embedding_dim = 64
        audio_config: dict[str, object] = {"sample_rate": 22050}

        def embed(self, path: Path) -> np.ndarray:
            # Use the digest as a seed for reproducible, finite float values
            digest = hashlib.sha256(str(path).encode()).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(64).astype(np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            return v

        def embed_batch(self, paths: list[Path]) -> list[np.ndarray]:
            return [self.embed(p) for p in paths]

    return MockEmbedder()
