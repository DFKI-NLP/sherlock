.PHONY: quality style test test-examples

quality:
	black --check --line-length 100 --target-version py37 tests sherlock scripts
	isort --check-only --recursive tests sherlock scripts
	flake8 tests sherlock scripts

# Format source code automatically

style:
	black --check --line-length 100 --target-version py37 tests sherlock scripts
	isort --check-only --recursive tests sherlock scripts

# Run tests for the library

test:
	python -m pytest -n auto --dist=loadfile -s -v ./tests/
