class CvRagError(Exception):
    """Base application error."""


class IndexBuildError(CvRagError):
    """Raised when the search index cannot be built."""


class LlmConfigError(CvRagError):
    """Raised when the LLM provider is not configured."""


class LlmResponseError(CvRagError):
    """Raised when the LLM provider returns an unusable response."""
