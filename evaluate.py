import pandas as pd
import numpy as np
from recommend import recommend

df_train = pd.read_excel('Gen_AI Dataset.xlsx', sheet_name='Train-Set')
train_groups = df_train.groupby('Query')['Assessment_url'].apply(list).to_dict()

def recall_at_k(recommendations, relevant, k=10):
    rec_urls = [r['url'] for r in recommendations[:k]]
    rel_set = set(relevant)
    rec_set = set(rec_urls)
    return len(rec_set & rel_set) / len(rel_set) if rel_set else 0

recalls = []
for query, relevants in train_groups.items():
    recs = recommend(query)
    recall = recall_at_k(recs, relevants)
    recalls.append(recall)
    print(f"Query: {query[:50]}... Recall@10: {recall:.2f}")

mean_recall = np.mean(recalls)
print(f"Mean Recall@10: {mean_recall:.2f}")