SPHINXBUILD ?= sphinx-build
PYTHON ?= python3

.PHONY: docs docs-fast serve_docs serve_docs-fast

docs:
	$(SPHINXBUILD) -b html docs docs/_build/html

docs-fast:
	FAST_BUILD=1 $(SPHINXBUILD) -b html docs docs/_build/html

serve_docs: docs
	$(PYTHON) -m http.server -d docs/_build/html

serve_docs-fast: docs-fast
	$(PYTHON) -m http.server -d docs/_build/html
