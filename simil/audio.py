"""Audio loading utilities for simil."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def load_clip(
    path: Path,
    sample_rate: int = 22050,
    max_duration: float = 180.0,
    start_pct: float = 0.10,
    mono: bool = True,
) -> np.ndarray:
    """Load an audio clip from a file.

    Uses ``librosa.get_duration`` to inspect the file without fully decoding it,
    then loads ``max_duration`` seconds starting at ``start_pct`` of total duration.
    If the file is shorter than ``max_duration``, the entire file is loaded from
    ``offset`` onward (librosa handles the truncation gracefully).

    Args:
        path: Path to the audio file.
        sample_rate: Target sample rate in Hz (resampled if necessary).
        max_duration: Maximum number of seconds to load.
        start_pct: Fraction of total duration to skip at the start (0.0–1.0).
            Useful to avoid intros/silence.
        mono: If True, mix down to a single channel.

    Returns:
        Float32 numpy array of audio samples, shape ``(n_samples,)`` if mono,
        ``(channels, n_samples)`` if not mono.
    """
    try:
        total_duration: float = librosa.get_duration(path=str(path))
    except Exception:
        logger.debug("Could not get duration for %s, loading from beginning", path)
        total_duration = max_duration

    offset = total_duration * start_pct if total_duration > 0.0 else 0.0

    logger.debug(
        "Loading clip from %s: offset=%.2fs, max_duration=%.2fs, sr=%d",
        path,
        offset,
        max_duration,
        sample_rate,
    )

    audio, _ = librosa.load(
        str(path),
        sr=sample_rate,
        mono=mono,
        offset=offset,
        duration=max_duration,
    )
    return audio.astype(np.float32)


def load_melspec(
    path: Path,
    sample_rate: int = 16000,
    n_mels: int = 128,
    hop_length: int = 512,
    max_duration: float = 180.0,
    start_pct: float = 0.10,
) -> np.ndarray:
    """Load an audio file and return its mel-spectrogram in dB scale.

    Args:
        path: Path to the audio file.
        sample_rate: Target sample rate in Hz.
        n_mels: Number of mel filter banks.
        hop_length: Hop length between STFT frames.
        max_duration: Maximum audio duration to load (seconds).
        start_pct: Fraction of total duration to skip at the start.

    Returns:
        Float32 numpy array of shape ``(n_mels, T)`` containing the
        mel-spectrogram values in dB (ref = max power).
    """
    audio = load_clip(
        path,
        sample_rate=sample_rate,
        max_duration=max_duration,
        start_pct=start_pct,
        mono=True,
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)
