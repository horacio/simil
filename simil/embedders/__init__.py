"""Embedder implementations and registry for simil.

The registry maps embedder names to their classes.  Use ``get_embedder()`` as
the single factory entry point — it handles lazy imports so that optional
dependencies (onnxruntime, laion-clap) are only required when the embedder is
actually used.

Adding a new embedder:
  1. Create a new module in ``simil/embedders/``.
  2. Subclass ``BaseEmbedder`` (and optionally implement ``TextEmbedder`` protocol).
  3. Add an entry to ``_REGISTRY`` below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from simil.core.exceptions import SimILError
from simil.embedders.base import BaseEmbedder
from simil.embedders.mfcc import MFCCEmbedder

if TYPE_CHECKING:
    from simil.embedders.clap import CLAPEmbedder
    from simil.embedders.effnet import EffNetEmbedder

__all__ = [
    "BaseEmbedder",
    "MFCCEmbedder",
    "get_embedder",
    "list_embedders",
]

# Registry: name → (module_path, class_name)
# Kept as strings to avoid importing optional heavy dependencies at module load.
_REGISTRY: dict[str, tuple[str, str]] = {
    "mfcc": ("simil.embedders.mfcc", "MFCCEmbedder"),
    "effnet-discogs": ("simil.embedders.effnet", "EffNetEmbedder"),
    "clap": ("simil.embedders.clap", "CLAPEmbedder"),
}


def get_embedder(name: str, **kwargs: object) -> BaseEmbedder:
    """Instantiate an embedder by name.

    Args:
        name: Embedder identifier, e.g. ``"mfcc"``, ``"effnet-discogs"``, ``"clap"``.
        **kwargs: Passed to the embedder's ``__init__``.

    Returns:
        A ready-to-use embedder instance.

    Raises:
        SimILError: If the name is not in the registry.
        EmbeddingError: If a required optional dependency is missing.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise SimILError(
            f"Unknown embedder {name!r}. Available: {available}"
        )
    module_path, class_name = _REGISTRY[name]
    import importlib  # noqa: PLC0415

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def list_embedders() -> list[str]:
    """Return the names of all registered embedders."""
    return sorted(_REGISTRY)


def register_embedder(name: str, module_path: str, class_name: str) -> None:
    """Register a custom embedder.

    Allows third-party code to add embedders without modifying this file.

    Args:
        name: Unique identifier for the embedder.
        module_path: Dotted Python module path, e.g. ``"mypackage.myembedder"``.
        class_name: Name of the class within that module.

    Example::

        from simil.embedders import register_embedder
        register_embedder("my-model", "mypackage.embedder", "MyEmbedder")
    """
    if name in _REGISTRY:
        raise SimILError(f"Embedder {name!r} is already registered.")
    _REGISTRY[name] = (module_path, class_name)
