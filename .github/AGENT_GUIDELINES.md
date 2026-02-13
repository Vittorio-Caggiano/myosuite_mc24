# Agent Guidelines

These guidelines are automatically loaded by the AI Issue Agent when creating PRs.
Edit this file to customize the agent's behavior for your repository.

## Change Policy
- **Minimal changes only**: Fix the issue with the smallest possible diff
- Do NOT refactor, reorganize, or "improve" code outside the scope of the issue
- Do NOT change formatting, style, or whitespace in lines you aren't fixing
- Preserve all existing comments, docstrings, and code structure

## Testing
- Add unit tests for every code change
- Place tests in the existing test directory structure
- Follow existing test naming conventions (e.g., `test_*.py`)
- Tests should verify the fix and prevent regression

## Code Style
- Follow the existing code style of the repository
- Do not introduce new dependencies unless absolutely necessary
- Keep backward compatibility — do not change public APIs
