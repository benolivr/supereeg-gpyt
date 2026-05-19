# -*- coding: utf-8 -*-
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath('../'))

# Mock GPU-only packages that can't be installed in the docs environment
for mod in ['cupy', 'cupy.cuda', 'cupy.cuda.compiler']:
    sys.modules[mod] = MagicMock()

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

# Never execute notebooks during docs build
nbsphinx_execute = 'never'
nbsphinx_allow_errors = False

# Generate API stubs (overwrite stale ones on each build)
autosummary_generate = True
autosummary_generate_overwrite = True

# numpydoc: don't generate per-method toctree entries (avoids "stub file not
# found" warnings for inherited/private methods that have no standalone page)
numpydoc_class_members_toctree = False

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = u'supereeg'
copyright = u'2017, Contextual Dynamics Laboratory'
author = u'Lucy L. W. Owen, Andrew C. Heusser, Jeremy R. Manning'

version = u'0.2'
release = u'0.2.1'

language = 'en'
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    # auto_examples: the pre-generated .rst files are canonical; exclude the
    # sphinx-gallery source files so Sphinx doesn't see duplicate documents.
    'auto_examples/*.py',
    'auto_examples/*.ipynb',
    'auto_examples/*.py.md5',
    # tutorial: the pre-generated .rst stubs are the canonical source; exclude
    # the raw .ipynb files so Sphinx doesn't try to process them as documents.
    'tutorial/*.ipynb',
]
pygments_style = 'sphinx'
todo_include_todos = False

# -- HTML output -----------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 3,
}

# -- HTMLHelp --------------------------------------------------------------

htmlhelp_basename = 'superEEGdoc'

# -- LaTeX output ----------------------------------------------------------

latex_elements = {}
latex_documents = [
    (master_doc, 'supereeg.tex', u'supereeg Documentation',
     u'Contextual Dynamics Laboratory', 'manual'),
]

# -- Man pages -------------------------------------------------------------

man_pages = [
    (master_doc, 'supereeg', u'supereeg Documentation', [author], 1)
]

# -- Texinfo output --------------------------------------------------------

texinfo_documents = [
    (master_doc, 'supereeg', u'supereeg Documentation',
     author, 'supereeg', 'One line description of project.',
     'Miscellaneous'),
]

