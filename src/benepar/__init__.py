"""
benepar: Berkeley Neural Parser
"""

# This file and all code in integrations/ relate to the version of the parser
# released via PyPI. If you only need to run research experiments, it is safe
# to delete the integrations/ folder and replace this __init__.py with an
# empty file.

__all__ = [
    "Parser",
    "InputSentence",
    "download",
    "BeneparComponent",
    "NonConstituentException",
]

from .integrations.downloader import download
from .integrations.nltk_plugin import Parser, InputSentence
from .integrations.spacy_plugin import BeneparComponent, NonConstituentException
