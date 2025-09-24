# Contributing to ITR (Instruction-Tool Retrieval)

Thank you for your interest in contributing to ITR! We welcome contributions from the community and are pleased to have you join us.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/uriafranko/ITR.git
   cd ITR
   ```

2. **Install dependencies:**
   ```bash
   uv sync --dev
   ```

3. **Install pre-commit hooks:**
   ```bash
   uv run pre-commit install
   ```

4. **Run tests to ensure everything is working:**
   ```bash
   uv run pytest
   ```

## 🔧 Development Workflow

### Making Changes

1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run the test suite:**
   ```bash
   uv run pytest
   ```

4. **Run code formatting and linting:**
   ```bash
   uv run black .
   uv run isort .
   uv run ruff check .
   ```

5. **Run type checking:**
   ```bash
   uv run mypy itr/
   ```

### Code Style

- We use [Black](https://black.readthedocs.io/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [Ruff](https://docs.astral.sh/ruff/) for linting
- [mypy](https://mypy.readthedocs.io/) for type checking
- Follow PEP 8 style guidelines
- Use type hints for all functions and methods
- Write docstrings for all public functions, classes, and modules

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `style:` formatting, missing semicolons, etc.
- `refactor:` code changes that neither fix bugs nor add features
- `test:` adding or updating tests
- `chore:` maintenance tasks

Examples:
```
feat: add budget-aware tool selection algorithm
fix: resolve token counting issue in chunking
docs: update installation instructions
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=itr --cov-report=html

# Run specific test file
uv run pytest tests/test_retrieval.py

# Run tests with specific markers
uv run pytest -m "not slow"
```

### Writing Tests

- Write tests for all new functionality
- Follow the existing test patterns and structure
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Mock external dependencies appropriately

## 📚 Documentation

- Update docstrings for any modified functions/classes
- Add examples for new features
- Update the README.md if needed
- Consider adding entries to the examples directory

## 🐛 Reporting Issues

When reporting issues, please include:

- Python version and operating system
- ITR version
- Complete error message and traceback
- Minimal code example that reproduces the issue
- Expected vs actual behavior

## 💡 Feature Requests

Before submitting a feature request:

1. Check existing issues to avoid duplicates
2. Clearly describe the use case and motivation
3. Provide examples of how the feature would be used
4. Consider implementation complexity and maintenance burden

## 📋 Pull Request Process

1. **Ensure your PR addresses an existing issue** or create one first
2. **Update tests** and documentation as needed
3. **Verify all checks pass** (tests, linting, type checking)
4. **Write a clear PR description** explaining:
   - What changes were made
   - Why they were necessary
   - How they were tested

### PR Checklist

- [ ] Tests pass locally
- [ ] Code is formatted with Black and isort
- [ ] Code passes linting (Ruff) and type checking (mypy)
- [ ] Documentation is updated
- [ ] Commit messages follow conventional format
- [ ] PR description clearly explains the changes

## 🤝 Code Review Process

- All PRs require at least one review from a maintainer
- Address all review comments before merging
- Maintain a respectful and constructive tone
- Be patient - reviews take time and help ensure code quality

## 🏷️ Release Process

Releases follow semantic versioning (SemVer):
- **Major**: Breaking changes
- **Minor**: New features (backwards compatible)
- **Patch**: Bug fixes (backwards compatible)

## 📞 Getting Help

- Open an issue for questions about contributing
- Join discussions in existing issues
- Reach out to maintainers for guidance on larger contributions

## 📜 Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment for everyone
- Report inappropriate behavior to project maintainers

## 🙏 Recognition

Contributors are recognized in:
- Release notes for significant contributions
- The project's contributor list
- Special mention for first-time contributors

Thank you for contributing to ITR! Your efforts help make this project better for everyone.
