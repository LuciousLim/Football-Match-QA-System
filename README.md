# Read me
## How to run
### 1. Set up venv and ipkernel
#### mac
```
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install jupyterlab ipykernel
python -m ipykernel install --user --name rag-spark-env --display-name "RAG"
```
#### windows
```
python -m venv .venv
.venv\Scripts\Activate

python.exe -m pip install --upgrade pip
pip install jupyterlab ipykernel
python -m ipykernel install --user --name rag-spark-env --display-name "RAG"
```

2. Open jupyter lab, choose RAG kernel. 
```
jupyter lab
```
## Requirements
```
pip install -r requirements.txt
```
