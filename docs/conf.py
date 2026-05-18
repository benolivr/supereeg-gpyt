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

# Generate API stubs
autosummary_generate = True

templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

project = u'supereeg'
copyright = u'2017, Contextual Dynamics Laboratory'
author = u'Lucy L. W. Owen, Andrew C. Heusser, Jeremy R. Manning'

version = u'0.2'
release = u'0.2.1'

language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
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

