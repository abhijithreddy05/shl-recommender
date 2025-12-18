import streamlit as st
import requests
import pandas as pd

st.title("SHL Assessment Recommender")

query = st.text_area("Enter query or JD:", height=100)
API_URL = "http://localhost:8000/recommend"  # Update to deployed API URL

if st.button("Get Recommendations"):
    if query:
        resp = requests.post(API_URL, json={"query": query})
        if resp.status_code == 200:
            data = resp.json()['recommended_assessments']
            df_out = pd.DataFrame(data)
            st.table(df_out[['name', 'url', 'duration_minutes', 'test_type']])
        else:
            st.error("Error: Check API.")
    else:
        st.warning("Enter a query.")