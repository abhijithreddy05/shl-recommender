import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import json
import google.generativeai as genai  # Old SDK

# Gemini setup
genai.configure(api_key="AIzaSyAiJdnHDPnCbTgqOmDDoMqWsPIDJf4CLxs")
model = genai.GenerativeModel('gemini-pro')  # Fixed: Free/stable model

df = pd.read_csv('shl_catalog_enriched.csv')
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load('embeddings.npy')

def recommend(query, top_k=10):
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
    
    # Rerank with Gemini
    candidate_list = candidates[['name', 'url', 'test_type', 'duration_minutes']].head(15).to_dict('records')
    prompt = f"""Rank these assessments for the query: "{query}".
Criteria: Relevance to skills (tech/soft), duration, balance (mix Knowledge & Skills + Personality if multi-domain).
Output JSON: {{"ranked": [{{"name": "name", "url": "url"}} for top {top_k}]}}"""
    try:
        response = model.generate_content(prompt + json.dumps(candidate_list))
        reranked_data = json.loads(response.text)['ranked']
        recs = []
        for r in reranked_data:
            match = candidates[candidates['name'] == r['name']]
            if not match.empty:
                recs.append(match.iloc[0][['name', 'url', 'adaptive_support', 'description', 'duration_minutes', 'remote_support', 'test_type']].to_dict())
        recs = recs[:top_k]
    except Exception as e:
        print(f"Rerank error: {e} – fallback to cosine")
        recs = candidates.head(top_k)[['name', 'url', 'adaptive_support', 'description', 'duration_minutes', 'remote_support', 'test_type']].to_dict('records')
    
    return recs

if __name__ == "__main__":
    query = "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes."
    recs = recommend(query)
    print(json.dumps({"recommended_assessments": recs}, indent=2))