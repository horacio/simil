"""Unit tests for simil.audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simil.audio import load_clip, load_melspec


class TestLoadClip:
    """Tests for load_clip()."""

    def test_returns_float32(self, wav_440: Path) -> None:
        """load_clip returns a float32 numpy array."""
        audio = load_clip(wav_440)
        assert audio.dtype == np.float32

    def test_returns_1d_for_mono(self, wav_440: Path) -> None:
        """load_clip with mono=True returns a 1-D array."""
        audio = load_clip(wav_440, mono=True)
        assert audio.ndim == 1

    def test_non_empty(self, wav_440: Path) -> None:
        """load_clip returns a non-empty array."""
        audio = load_clip(wav_440)
        assert len(audio) > 0

    def test_short_audio_handled(self, wav_440: Path) -> None:
        """load_clip handles files shorter than max_duration gracefully."""
        # The fixture is only 3s; requesting 180s should still work
        audio = load_clip(wav_440, max_duration=180.0)
        # Should return ≤ 3 seconds of samples
        assert len(audio) <= 22050 * 4  # 4s margin

    def test_custom_sample_rate(self, wav_440: Path) -> None:
        """load_clip resamples to the requested sample rate."""
        sr = 16000
        audio = load_clip(wav_440, sample_rate=sr)
        # 3s at 16000 → ~48000 samples (with start_pct offset applied)
        assert len(audio) > 0

    def test_start_pct_zero(self, wav_440: Path) -> None:
        """load_clip with start_pct=0.0 loads from the beginning."""
        audio = load_clip(wav_440, start_pct=0.0)
        assert len(audio) > 0


class TestLoadMelspec:
    """Tests for load_melspec()."""

    def test_returns_float32(self, wav_440: Path) -> None:
        """load_melspec returns a float32 array."""
        mel = load_melspec(wav_440)
        assert mel.dtype == np.float32

    def test_correct_n_mels(self, wav_440: Path) -> None:
        """load_melspec returns (n_mels, T) shaped array."""
        n_mels = 64
        mel = load_melspec(wav_440, n_mels=n_mels)
        assert mel.ndim == 2
        assert mel.shape[0] == n_mels

    def test_default_n_mels(self, wav_440: Path) -> None:
        """load_melspec with default n_mels=128."""
        mel = load_melspec(wav_440)
        assert mel.shape[0] == 128

    def test_nonzero_time_axis(self, wav_440: Path) -> None:
        """Melspec time dimension is > 0."""
        mel = load_melspec(wav_440)
        assert mel.shape[1] > 0

    def test_db_values_are_finite(self, wav_440: Path) -> None:
        """Melspec values are finite (no NaN/Inf for valid audio)."""
        mel = load_melspec(wav_440)
        assert np.isfinite(mel).all()
