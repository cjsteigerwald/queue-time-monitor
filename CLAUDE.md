# CLAUDE.md

## Git conventions

- Never include "Co-Authored-By: Claude" or any Claude/AI attribution in commit messages, PR descriptions, or any git-related text.
- When fixing a GitHub issue, always create a feature branch for the work. Once complete, commit and create a PR back into main.
- Before creating a PR, run the `code-reviewer` agent against the changes and address any Critical or Warning findings first.

## Commit conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/) enforced by commitizen and pre-commit hooks.

**Format:** `type(scope): subject`

**Types:** `feat`, `fix`, `refactor`, `chore`, `test`, `docs`, `ci`

**Scopes (optional):** `detection`, `estimation`, `web`, `storage`, `config`, `cli`

**Breaking changes:** `type(scope)!: subject`

**Examples:**
- `feat(web): add real-time queue heatmap overlay`
- `fix(storage): parameterize SQL query to prevent injection`
- `chore: upgrade ultralytics to 8.1`
- `test(detection): add edge case tests for empty frames`

**Versioning:** Run `cz bump` to auto-determine semver bump, update changelog, and create a git tag.
