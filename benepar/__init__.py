"""
benepar: Berkeley Neural Parser
"""

from .downloader import download
from .nltk_plugin import Parser

__all__ = ['Parser', 'download']
