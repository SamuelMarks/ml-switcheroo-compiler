.PHONY: docs docs-fast serve_docs serve_docs-fast

docs:
	.venv/bin/sphinx-build -b html docs docs/_build/html

docs-fast:
	FAST_BUILD=1 .venv/bin/sphinx-build -b html docs docs/_build/html

serve_docs: docs
	python3 -m http.server -d docs/_build/html

serve_docs-fast: docs-fast
	python3 -m http.server -d docs/_build/html
