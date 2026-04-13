## Installation

```bash
python -m venv --system-site-packages  ~/jgould/venv
source ~/jgould/venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

## Dependency management

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