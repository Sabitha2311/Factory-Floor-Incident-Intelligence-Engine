import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import json
from datetime import datetime

# Import custom pipelines
from ml_pipeline import preprocess_text
from llm_pipeline import LLMPipeline

st.set_page_config(page_title="Factory Floor Incident Intelligence", layout="wide")

# --- Load Models & Data ---
@st.cache_resource
def load_ml_models():
    if os.path.exists("models/tfidf_vectorizer.pkl"):
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        clf_type = joblib.load("models/type_classifier.pkl")
        clf_sev = joblib.load("models/severity_classifier.pkl")
        return vectorizer, clf_type, clf_sev
    return None, None, None

@st.cache_resource
def load_llm_pipeline():
    return LLMPipeline()

@st.cache_data
def load_data():
    if os.path.exists("data/incident_reports.csv"):
        df = pd.read_csv("data/incident_reports.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

vectorizer, clf_type, clf_sev = load_ml_models()
llm_pipe = load_llm_pipeline()
df = load_data()

# Build vector DB if empty
if llm_pipe.collection.count() == 0 and not df.empty:
    with st.spinner("Initializing Vector Database..."):
        llm_pipe.build_vector_db(df)

# --- UI Layout ---
st.title("🏭 Factory Floor Incident Intelligence Engine")

tab1, tab2 = st.tabs(["Real-Time Triage & Extraction", "Management Dashboard"])

with tab1:
    st.header("Incident Triage & Analysis")
    
    # Model configuration
    with st.sidebar:
        st.subheader("LLM Configuration")
        llm_model_name = st.selectbox("LLM Model", ["gemini-2.5-flash", "gpt-4o-mini", "gpt-4o"])
        api_key = st.text_input("API Key (Optional if set in ENV)", type="password")
        if api_key:
            if "gpt" in llm_model_name:
                os.environ["OPENAI_API_KEY"] = api_key
            elif "gemini" in llm_model_name:
                os.environ["GEMINI_API_KEY"] = api_key
                
    st.markdown("Enter a new raw incident report below to automatically classify it, predict severity, and suggest root causes.")
    
    query = st.text_area("Incident Report Text", "The conveyor belt motor on M-102 is vibrating heavily and making a loud grinding sound.", height=150)
    
    if st.button("Analyze Incident", type="primary"):
        if not vectorizer:
            st.error("ML Models not found. Please run `python src/ml_pipeline.py` first.")
        else:
            col1, col2 = st.columns(2)
            
            with st.spinner("Classifying text..."):
                # 1. ML Classification
                proc_text = preprocess_text(query)
                vec_text = vectorizer.transform([proc_text])
                
                pred_type = clf_type.predict(vec_text)[0]
                pred_sev = clf_sev.predict(vec_text)[0]
                
                with col1:
                    st.subheader("ML Predictions")
                    st.metric("Predicted Incident Type", pred_type)
                    st.metric("Predicted Severity", pred_sev)
            
            with st.spinner("Running Semantic Search & LLM Extraction..."):
                # 2. Semantic Search
                similar = llm_pipe.search_similar_incidents(query, top_k=3)
                
                # 3. LLM Extraction
                result = llm_pipe.extract_and_summarize(query, similar, llm_model=llm_model_name)
                
                with col2:
                    st.subheader("LLM Extraction & Action Plan")
                    if "error" in result:
                        st.error(f"LLM Error: {result['message']}")
                        st.text(result['error'])
                    else:
                        st.json(result)
                        
            st.divider()
            st.subheader("Historically Similar Incidents (RAG Context)")
            for i, sim in enumerate(similar):
                with st.expander(f"Match {i+1}: {sim['metadata']['machine_id']} ({sim['metadata']['incident_type']}) - Score: {sim['distance']:.2f}"):
                    st.write(f"**Text:** {sim['document']}")
                    st.write(f"**Root Cause:** {sim['metadata']['root_cause']}")
                    st.write(f"**Severity:** {sim['metadata']['severity']}")

with tab2:
    st.header("Management Analytics")
    if df.empty:
        st.warning("No data found. Please run the data generator.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Incidents", len(df))
        col2.metric("High/Critical Severity", len(df[df['severity'].isin(['High', 'Critical'])]))
        col3.metric("Total Downtime (Hrs)", round(df['downtime_minutes'].sum() / 60))
        col4.metric("Active Machines", df['machine_id'].nunique())
        
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Incidents by Type")
            fig_type = px.pie(df, names="incident_type", hole=0.4)
            st.plotly_chart(fig_type, use_container_width=True)
            
        with c2:
            st.subheader("Severity Distribution")
            fig_sev = px.bar(df['severity'].value_counts().reset_index(), x='severity', y='count', color='severity')
            st.plotly_chart(fig_sev, use_container_width=True)
            
        st.subheader("High Severity Escalation Queue")
        st.dataframe(
            df[df['severity'].isin(['High', 'Critical'])][['timestamp', 'machine_id', 'incident_type', 'incident_text', 'root_cause']]
            .sort_values('timestamp', ascending=False)
            .head(10),
            use_container_width=True
        )
