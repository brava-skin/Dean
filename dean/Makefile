.PHONY: help install install-dev test lint format clean run

help:
	@echo "Dean - Enterprise Meta Ads Automation"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test         - Run test suite"
	@echo "  make lint         - Run linters (mypy, ruff)"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make run          - Run Dean automation"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black mypy ruff

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	mypy src/
	ruff check src/

format:
	black src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage

run:
	python src/main.py

