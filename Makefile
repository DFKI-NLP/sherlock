.PHONY: quality style test test-examples

quality:
	black --check --line-length 100 --target-version py37 tests sherlock scripts
	isort --check-only --recursive --verbose tests sherlock scripts
	mypy sherlock --ignore-missing-imports
	flake8 tests sherlock scripts

# Format source code automatically

style:
	black --line-length 100 --target-version py37 tests sherlock scripts
	isort --recursive tests sherlock scripts

# Run tests for the library

test:
	python -m pytest -n 1 --dist=loadfile -s -v ./tests/
