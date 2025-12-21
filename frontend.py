import streamlit as st
import requests
import pandas as pd

st.title("SHL Assessment Recommender")

query = st.text_area("Enter query or JD:", height=100)
API_URL = "https://shl-recommender-8qdx.onrender.com/recommend"  # Your Render URL (no trailing slash)

if st.button("Get Recommendations"):
    if query:
        headers = {'Content-Type': 'application/json', 'User-Agent': 'Streamlit'}  # Headers for CORS
        try:
            with st.spinner("Fetching... (first time slow – Render cold start)"):
                resp = requests.post(API_URL, json={"query": query}, headers=headers, timeout=120)  # 2 min timeout
            if resp.status_code == 200:
                data = resp.json()['recommended_assessments']
                df_out = pd.DataFrame(data)
                st.table(df_out[['name', 'url', 'duration_minutes', 'test_type']])
            else:
                st.error(f"API Error {resp.status_code}: {resp.text[:200]}... Check Render logs.")
        except requests.exceptions.Timeout:
            st.error("Timeout – Refresh and try again (Render waking up).")
        except Exception as e:
            st.error(f"Connection error: {e}. Check API URL/CORS.")
    else:
        st.warning("Enter a query.")