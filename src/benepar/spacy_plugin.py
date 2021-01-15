__all__ = ["BeneparComponent", "NonConstituentException"]

import warnings

from .integrations.spacy_plugin import BeneparComponent, NonConstituentException

warnings.warn(
    "BeneparComponent and NonConstituentException have been moved to the benepar "
    "module. Use `from benepar import BeneparComponent, NonConstituentException` "
    "instead of benepar.spacy_plugin. The benepar.spacy_plugin namespace is deprecated "
    "and will be removed in a future version.",
    FutureWarning,
)
