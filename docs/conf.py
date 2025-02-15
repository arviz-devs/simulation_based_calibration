# ruff: noqa
# -*- coding: utf-8 -*-
import os
import sys

from simuk import __version__

sys.path.insert(0, os.path.abspath("../simuk"))

# -- Project information -----------------------------------------------------

project = "Simuk"
author = "ArviZ contributors"
copyright = f"2025, {author}"

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_nb",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "jupyter_sphinx",
]

# -- Extension configuration -------------------------------------------------
nb_execution_mode = "auto"
nb_execution_excludepatterns = ["*.ipynb"]
nb_kernel_rgx_aliases = {".*": "python3"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_flags = ["members"]

# autodoc_default_options = {
#     "inherited-members": True,
# }

source_suffix = ".rst"

master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
pygments_style = "sphinx"
html_static_path = ["_static", "logos", "audios"]
html_css_files = ["custom.css"]
html_theme = "pydata_sphinx_theme"
html_favicon = "logos/favicon.ico"

html_theme_options = {
    "collapse_navigation": True,
    "show_toc_level": 2,
    "navigation_depth": 4,
    "search_bar_text": "Search the docs...",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/arviz-devs/simuk",
            "icon": "fa-brands fa-github",
        },
    ],
    # "logo": {
    #     "image_light": "Simuk_flat.png",
    #     "image_dark": "Simuk_flat_white.png",
    # },
}
html_context = {
    "github_user": "arviz-devs",
    "github_repo": "simuk",
    "github_version": "main",
    "doc_path": "docs/",
    "default_mode": "light",
}


# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "simukdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, "simuk.tex", "simuk Documentation", "The developers of simuk", "manual"),
]


# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "simuk", "simuk Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "simuk",
        "simuk Documentation",
        author,
        "simuk",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

epub_exclude_files = ["search.html"]
