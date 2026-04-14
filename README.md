## Installation
Public-facing install process:
```bash
git clone https://github.com/draeloslab/cl_project.git
cd cl_project

python -m venv --system-site-packages env  # the --system-site-packages is necessary on CL1 instances
source venv/bin/activate
pip install -e ".[dev]"
python -m ipykernel install --user --name=cl_project_venv
nbstripout --install
```

To document what I ususally do:
```bash
deactivate # if a venv is already active
cd ~/jgould/

git clone git@github.com:draeloslab/cl_project.git
cd cl_project
git remote set-url origin https://github.com/draeloslab/cl_project.git
git remote set-url --push origin git@github.com:draeloslab/cl_project.git
git config user.name "Jonathan Gould"
git config user.email "jonathan.d.gould@gmail.com"
git config alias.l "log --color --graph --oneline --abbrev-commit --max-count=10"


python -m venv --system-site-packages ~/jgould/cl_project/venv
source ~/jgould/cl_project/venv/bin/activate

pip install -e ".[dev]"
python -m ipykernel install --user --name=cl_project_venv
nbstripout --install
ln -s /home/labuser/jgould/cl_project/workspace /home/labuser/notebooks
```

## Uninstall

```bash
cd ~/jgould/
rm -rf cl_project
jupyter kernelspec list
jupyter kernelspec uninstall unwanted-kernel
rm /home/labuser/notebooks/workspace
```

```bash
pip freeze > requirements-lock.txt
```
