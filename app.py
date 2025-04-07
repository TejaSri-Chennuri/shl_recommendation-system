import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re

# Load data
try:
    df = pd.read_csv("shl_assessments.csv")
except FileNotFoundError:
    st.error("Error: 'shl_assessments.csv' not found. Please ensure it's in the same folder.")
    st.stop()

df["Text"] = df["Assessment Name"] + " " + df["Description"] + " " + df["Test Type"]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["Text"])

def extract_duration(query):
    match = re.search(r"(\d+)\s*(minutes|mins)", query, re.IGNORECASE)
    return int(match.group(1)) if match else None

def recommend_assessments(query):
    duration = extract_duration(query)
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-10:][::-1]
    result = df.iloc[top_indices].copy()
    if duration:
        result = result[result["Duration (min)"] <= duration]
    if result.empty:
        return df.iloc[[similarities.argmax()]]
    return result.head(10)

# Streamlit UI
st.title("SHL Assessment Recommender")
query = st.text_input("Enter job description or query:", "")
if query:
    recommendations = recommend_assessments(query)
    st.table(recommendations[["Assessment Name", "URL", "Remote Testing Support", 
                              "Adaptive/IRT Support", "Duration (min)", "Test Type"]])

# API-like JSON output
def get_api_result(query):
    recommendations = recommend_assessments(query)
    return recommendations.to_json(orient="records")

if st.button("Get API Result"):
    st.json(get_api_result(query))