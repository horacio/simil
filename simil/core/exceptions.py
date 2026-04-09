"""Exception hierarchy for simil."""

from __future__ import annotations


class SimILError(Exception):
    """Base exception for all simil errors."""


class EmbeddingError(SimILError):
    """Raised when audio embedding fails."""


class ResolverError(SimILError):
    """Raised when a URL or source cannot be resolved."""


class UnsupportedURLError(ResolverError):
    """Raised when a URL scheme or host is not supported."""


class SpotifyPreviewUnavailableError(ResolverError):
    """Raised when a Spotify track preview is not available."""


class SimILIndexError(SimILError):
    """Base class for index-related errors.

    Named SimILIndexError to avoid shadowing Python's built-in IndexError.
    """


class IndexSchemaError(SimILIndexError):
    """Raised when the index schema_version does not match the expected version."""


class IndexDimensionError(SimILIndexError):
    """Raised when a vector's dimension does not match the index embedding_dim."""


class IndexEmbedderMismatch(SimILIndexError):
    """Raised when the embedder name in meta.json differs from the current embedder.

    Run with --full to force a complete rebuild.
    """


class LibraryError(SimILError):
    """Raised for file-system or library scanning errors."""


class CatalogError(SimILError):
    """Base class for catalog-related errors."""


class CatalogSchemaError(CatalogError):
    """Raised when catalog.json schema_version does not match the expected version."""
