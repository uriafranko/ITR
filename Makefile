# ITR Project Makefile
# Professional development workflow automation

.PHONY: help install install-dev test test-unit test-integration test-coverage clean lint format type-check security docs build publish pre-commit setup-hooks run-hooks validate all-checks dev-setup production-setup

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

# Project configuration
PACKAGE_NAME := itr
PYTHON_VERSION := 3.8
UV_PYTHON := uv run python
UV_PIP := uv add
TEST_PATH := tests/
SRC_PATH := $(PACKAGE_NAME)/
DOCS_PATH := docs/
BUILD_PATH := dist/
COVERAGE_THRESHOLD := 80

help: ## Show this help message
	@echo "$(CYAN)ITR Development Makefile$(RESET)"
	@echo "$(YELLOW)Available targets:$(RESET)"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo
	@echo "$(YELLOW)Quick start:$(RESET)"
	@echo "  make dev-setup    # Set up development environment"
	@echo "  make test         # Run all tests"
	@echo "  make lint         # Check code quality"
	@echo "  make all-checks   # Run all quality checks"

## Installation and Setup
install: ## Install package dependencies
	@echo "$(BLUE)Installing package dependencies...$(RESET)"
	uv sync

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	uv sync --dev

dev-setup: install-dev setup-hooks ## Complete development environment setup
	@echo "$(GREEN)Development environment ready!$(RESET)"
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  - Run 'make test' to verify everything works"
	@echo "  - Run 'make lint' to check code quality"
	@echo "  - Check README.md for usage examples"

production-setup: install ## Production environment setup
	@echo "$(GREEN)Production environment ready!$(RESET)"

## Testing
test: test-unit test-integration ## Run all tests
	@echo "$(GREEN)All tests completed!$(RESET)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(UV_PYTHON) -m pytest $(TEST_PATH)unit/ -v --tb=short

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(UV_PYTHON) -m pytest $(TEST_PATH)integration/ -v --tb=short

test-fast: ## Run tests with minimal output
	@echo "$(BLUE)Running fast tests...$(RESET)"
	$(UV_PYTHON) -m pytest $(TEST_PATH) -x -q

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(UV_PYTHON) -m pytest $(TEST_PATH)
	@echo "$(GREEN)Coverage report generated in htmlcov/$(RESET)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode (Ctrl+C to stop)...$(RESET)"
	$(UV_PYTHON) -m pytest $(TEST_PATH) --tb=short -f

test-performance: ## Run performance/slow tests
	@echo "$(BLUE)Running performance tests...$(RESET)"
	$(UV_PYTHON) -m pytest $(TEST_PATH) -m "slow" -v

test-specific: ## Run specific test (usage: make test-specific TEST=test_name)
	@echo "$(BLUE)Running specific test: $(TEST)$(RESET)"
	$(UV_PYTHON) -m pytest $(TEST_PATH) -k "$(TEST)" -v

## Code Quality
lint: ## Run all linting checks
	@echo "$(BLUE)Running linting checks...$(RESET)"
	$(UV_PYTHON) -m ruff check $(SRC_PATH) $(TEST_PATH)
	$(UV_PYTHON) -m ruff format --check $(SRC_PATH) $(TEST_PATH)
	@echo "$(GREEN)Linting completed!$(RESET)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(UV_PYTHON) -m black $(SRC_PATH) $(TEST_PATH)
	$(UV_PYTHON) -m isort $(SRC_PATH) $(TEST_PATH)
	$(UV_PYTHON) -m ruff format $(SRC_PATH) $(TEST_PATH)
	@echo "$(GREEN)Code formatted!$(RESET)"

format-check: ## Check if code is properly formatted
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	$(UV_PYTHON) -m black --check $(SRC_PATH) $(TEST_PATH)
	$(UV_PYTHON) -m isort --check-only $(SRC_PATH) $(TEST_PATH)

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(RESET)"
	$(UV_PYTHON) -m mypy $(SRC_PATH) --show-error-codes
	@echo "$(GREEN)Type checking completed!$(RESET)"

type-check-strict: ## Run strict type checking
	@echo "$(BLUE)Running strict type checks...$(RESET)"
	$(UV_PYTHON) -m mypy $(SRC_PATH) --strict --show-error-codes

security: ## Run security checks with bandit
	@echo "$(BLUE)Running security checks...$(RESET)"
	$(UV_PYTHON) -m bandit -r $(SRC_PATH) -f json
	@echo "$(GREEN)Security checks completed!$(RESET)"

validate: lint type-check security ## Run all validation checks
	@echo "$(GREEN)All validation checks passed!$(RESET)"

all-checks: test-coverage validate ## Run all quality checks including tests
	@echo "$(GREEN)🎉 All checks passed! Ready for production.$(RESET)"

## Pre-commit Hooks
setup-hooks: ## Install pre-commit hooks
	@echo "$(BLUE)Setting up pre-commit hooks...$(RESET)"
	$(UV_PYTHON) -m pre_commit install
	@echo "$(GREEN)Pre-commit hooks installed!$(RESET)"

run-hooks: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	$(UV_PYTHON) -m pre_commit run --all-files

update-hooks: ## Update pre-commit hooks
	@echo "$(BLUE)Updating pre-commit hooks...$(RESET)"
	$(UV_PYTHON) -m pre_commit autoupdate


## Building and Publishing
build: clean ## Build package for distribution
	@echo "$(BLUE)Building package...$(RESET)"
	$(UV_PYTHON) -m build
	@echo "$(GREEN)Package built in $(BUILD_PATH)$(RESET)"

publish-test: build ## Publish to TestPyPI
	@echo "$(BLUE)Publishing to TestPyPI...$(RESET)"
	$(UV_PYTHON) -m twine upload --repository testpypi $(BUILD_PATH)/*
	@echo "$(GREEN)Published to TestPyPI!$(RESET)"

publish: build ## Publish to PyPI
	@echo "$(RED)Publishing to PyPI...$(RESET)"
	@read -p "Are you sure you want to publish to PyPI? (y/N): " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(UV_PYTHON) -m twine upload $(BUILD_PATH)/* && \
		echo "$(GREEN)Published to PyPI!$(RESET)"; \
	else \
		echo "$(YELLOW)Publish cancelled.$(RESET)"; \
	fi

## Development Utilities
clean: ## Clean build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf $(BUILD_PATH)
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Cleaned!$(RESET)"

reset: clean ## Reset development environment
	@echo "$(BLUE)Resetting development environment...$(RESET)"
	rm -rf .venv/
	uv venv
	make install-dev

## Performance and Profiling
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(RESET)"
	@$(UV_PYTHON) -c "import time; from itr import ITR, ITRConfig; config = ITRConfig(); itr = ITR(config); itr.add_instruction('Test instruction for benchmarking.'); start = time.time(); [itr.step('benchmark query') for i in range(100)]; end = time.time(); print(f'100 retrievals took {end-start:.3f}s ({(end-start)*10:.1f}ms per retrieval)')"
	@echo "$(GREEN)Benchmarks completed!$(RESET)"

profile: ## Profile code performance
	@echo "$(BLUE)Profiling performance...$(RESET)"
	$(UV_PYTHON) -m cProfile -s cumulative -m pytest $(TEST_PATH) -x

## Configuration and Environment
config-validate: ## Validate configuration files
	@echo "$(BLUE)Validating configuration...$(RESET)"
	@$(UV_PYTHON) -c "from itr.core.config import ITRConfig; import sys; config = ITRConfig(); config.validate(); print('✅ Default configuration is valid')" || (echo "❌ Configuration error" && exit 1)

env-info: ## Show environment information
	@echo "$(CYAN)Environment Information:$(RESET)"
	@echo "Python version: $$($(UV_PYTHON) --version)"
	@echo "UV version: $$(uv --version)"
	@echo "Working directory: $$(pwd)"
	@echo "Package version: $$($(UV_PYTHON) -c 'from itr import __version__; print(__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "Dependencies:"
	@uv tree --depth 1

dependency-check: ## Check for dependency issues
	@echo "$(BLUE)Checking dependencies...$(RESET)"
	uv sync --frozen
	@echo "$(GREEN)Dependencies are consistent!$(RESET)"

## CI/CD Simulation
ci-test: ## Run tests as in CI
	@echo "$(BLUE)Running CI test simulation...$(RESET)"
	make clean
	make install-dev
	make test-coverage
	make validate
	@echo "$(GREEN)CI tests completed!$(RESET)"

pre-push: all-checks ## Run before pushing to repository
	@echo "$(GREEN)✅ Ready to push!$(RESET)"

release-check: ## Check if ready for release
	@echo "$(BLUE)Checking release readiness...$(RESET)"
	make all-checks
	make build
	@echo "$(GREEN)✅ Ready for release!$(RESET)"

## Help and Information
version: ## Show package version
	@$(UV_PYTHON) -c "from itr import __version__; print(f'ITR version: {__version__}')" 2>/dev/null || echo "Package not installed"

status: ## Show project status
	@echo "$(CYAN)ITR Project Status:$(RESET)"
	@echo "Git status:"
	@git status --porcelain || echo "Not a git repository"
	@echo "\nRecent commits:"
	@git log --oneline -5 2>/dev/null || echo "No git history"
	@echo "\nDevelopment environment:"
	@make env-info

todo: ## Show project TODOs
	@echo "$(CYAN)Project TODOs:$(RESET)"
	@grep -r "TODO\|FIXME\|XXX" $(SRC_PATH) $(TEST_PATH) || echo "No TODOs found!"

stats: ## Show project statistics
	@echo "$(CYAN)Project Statistics:$(RESET)"
	@echo "Lines of code:"
	@find $(SRC_PATH) -name "*.py" | xargs wc -l | tail -1
	@echo "Test files:"
	@find $(TEST_PATH) -name "*.py" | wc -l | sed 's/^/  /'
	@echo "Documentation files:"
	@find . -name "*.md" -o -name "*.rst" | wc -l | sed 's/^/  /'
