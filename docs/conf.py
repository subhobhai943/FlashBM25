from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "FlashBM25"
author = "FlashBM25 contributors"
copyright = "2026, FlashBM25 contributors"

try:
    release = metadata.version("flashbm25")
except metadata.PackageNotFoundError:
    release = "0.2.0"

version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_mock_imports = ["flashbm25._flashbm25"]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autosummary_generate = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
}

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_title = "FlashBM25 documentation"
