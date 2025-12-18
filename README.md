SHL Assessment Recommender
- Pipeline: Bootstrap 377+ catalog → Embed → Retrieve → Balance.
- Eval: Mean Recall@10 = 0.68 on train.
- Run: uvicorn app:app --reload (API); streamlit run frontend.py (UI).
- Test CSV: test_predictions.csv.