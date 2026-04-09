## Installation

```bash
# From the project root – installs cl_project plus dev tools (pytest)
pip install -e ".[dev]"
```

## Dependency management

`pyproject.toml` uses lower-bound constraints (e.g. `numpy>=2.4`) to stay
permissive while preventing known-incompatible older versions.

`requirements-lock.txt` records the exact versions of the last known-good
environment. If a fresh install breaks due to a conflicting upgrade, restore it
with:

```bash
pip install -r requirements-lock.txt
pip install -e ".[dev]"
```

To update the lock file after verifying a new working environment:

```bash
pip freeze > requirements-lock.txt
```