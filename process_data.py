import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

df = pd.read_csv('shl_catalog_enriched.csv')  # Changed here
embedder = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = df['description'].tolist()
embeddings = embedder.encode(descriptions)
np.save('embeddings.npy', embeddings)
df['embedding'] = embeddings.tolist()
df.to_csv('shl_catalog_embedded.csv', index=False)
print("Embeddings updated with enriched descs.")