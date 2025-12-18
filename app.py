from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import json

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Lazy load globals (OOM fix – loads on first request)
_df = None
_embedder = None
_embeddings = None

def load_data():
    global _df, _embedder, _embeddings
    if _df is None:
        _df = pd.read_csv('shl_catalog_enriched.csv')
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        _embeddings = _embedder.encode(_df['description'].tolist())  # Regenerate – no npy needed
    return _df, _embedder, _embeddings

def recommend(query, top_k=10):
    df, embedder, embeddings = load_data()
    query_emb = embedder.encode([query])
    sims = cosine_similarity(query_emb, embeddings).flatten()
    top_indices = np.argsort(sims)[-top_k*3:][::-1]
    
    candidates = df.iloc[top_indices].copy()
    candidates['similarity'] = sims[top_indices]
    
    # Duration filter
    dur_match = re.search(r'(\d+)( minutes?| hours?)', query.lower())
    if dur_match:
        dur = int(dur_match.group(1)) * (60 if 'hour' in dur_match.group(2) else 1)
        candidates = candidates[candidates['duration_minutes'] <= dur]
    
    # Balance (mix types for multi-domain)
    if any(word in query.lower() for word in ['collaborate', 'communication', 'personality']):
        ks = candidates[candidates['test_type'].str.contains('Knowledge', na=False)]
        pb = candidates[candidates['test_type'].str.contains('Personality', na=False)]
        balanced = pd.concat([ks.head(5), pb.head(5)]).drop_duplicates().head(top_k)
    else:
        balanced = candidates.head(top_k)
    
    recs = balanced[['name', 'url', 'adaptive_support', 'description', 'duration_minutes', 'remote_support', 'test_type']].to_dict('records')
    return recs

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def get_recommend(req: QueryRequest):
    recs = recommend(req.query, top_k=5)
    if not recs:
        raise HTTPException(status_code=404, detail="No recommendations found")
    return {"recommended_assessments": recs}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)