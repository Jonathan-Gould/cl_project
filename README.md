## Installation

```bash
cd ~/jgould/
git clone https://github.com/draeloslab/cl_project
cd cl_project
python -m venv --system-site-packages ~/jgould/cl_project/venv
source ~/jgould/cl_project/venv/bin/activate

pip install -e ".[dev]"
python -m ipykernel install --user --name=venv
```

## Uninstall

```bash
cd ~/jgould/
rm -rf cl_project
jupyter kernelspec list
jupyter kernelspec uninstall unwanted-kernel
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

