# Read me
## How to run
1. Set up venv and ipkernel
```
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install jupyterlab ipykernel
python -m ipykernel install --user --name rag-spark-env --display-name "RAG+Spark"
```
2. Open jupyter lab, choose RAG+Spark kernel. 
```
jupyter lab
```
## Requirements
```
pip install -r requirements.txt
```
