from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

_df = None
_embedder = None
_embeddings = None

def load_data():
    global _df, _embedder, _embeddings
    if _df is None:
        logger.info("Loading dataset...")
        _df = pd.read_csv('shl_catalog_enriched.csv')
        logger.info("Loading lighter model...")
        _embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # 50MB model
        logger.info("Loading pre-computed embeddings...")
        _embeddings = np.load('embeddings.npy')  # Pre-computed – fast
        logger.info("Load complete.")
    return _df, _embedder, _embeddings

def recommend(query, top_k=10):
    try:
        df, embedder, embeddings = load_data()
        logger.info(f"Processing query: {query[:50]}...")
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
        
        # Balance
        if any(word in query.lower() for word in ['collaborate', 'communication', 'personality']):
            ks = candidates[candidates['test_type'].str.contains('Knowledge', na=False)]
            pb = candidates[candidates['test_type'].str.contains('Personality', na=False)]
            balanced = pd.concat([ks.head(5), pb.head(5)]).drop_duplicates().head(top_k)
        else:
            balanced = candidates.head(top_k)
        
        recs = balanced[['name', 'url', 'adaptive_support', 'description', 'duration_minutes', 'remote_support', 'test_type']].to_dict('records')
        logger.info(f"Generated {len(recs)} recs.")
        return recs
    except Exception as e:
        logger.error(f"Recommend error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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