import pandas as pd
from recommend import recommend  # Assumes your recommend.py is in the same dir

# Load test queries from Excel (skip header row 1)
df_test = pd.read_excel('Gen_AI Dataset.xlsx', sheet_name='Test-Set')
queries = df_test['Query'].tolist()[1:]  # 9 queries

predictions = []
for q_id, query in enumerate(queries, 1):
    recs = recommend(query, top_k=5)  # 5 recs per query
    for rec in recs:
        predictions.append({'Query': f"Query {q_id}", 'Assessment_url': rec['url']})

df_pred = pd.DataFrame(predictions)
df_pred.to_csv('test_predictions.csv', index=False)
print("Generated test_predictions.csv – 45 rows ready for submission!")