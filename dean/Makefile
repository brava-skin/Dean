.PHONY: help install install-dev lint format clean run

help:
	@echo "Dean - Enterprise Meta Ads Automation"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make lint         - Run linters (mypy, ruff)"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make run          - Run Dean automation"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install black mypy ruff

lint:
	mypy src/
	ruff check src/

format:
	black src/

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	rm -rf build/ dist/ .mypy_cache/

run:
	python src/main.py

