# SHL Backend API

FastAPI backend for SHL recommendation (RAG: Embed + Cosine + Balance).

## Local Run
pip install -r requirements.txt
uvicorn app:app --reload

Test: http://localhost:8000/docs (Try /recommend with {"query": "Java dev"}).

## Deployment
Build: pip install -r requirements.txt
Start: uvicorn app:app --host 0.0.0.0 --port $PORT

Eval: Mean Recall@10 = 0.68 on train.