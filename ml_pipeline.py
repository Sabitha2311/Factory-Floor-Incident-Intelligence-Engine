import pandas as pd
import numpy as np
import os
import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import lightgbm as lgb

# Load spaCy model for lemmatization (optional, but good for NLP)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy en_core_web_sm model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Basic lemmatization and lowercasing
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def train_and_evaluate_models(data_path="data/incident_reports.csv", model_dir="models/"):
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing text data...")
    df["processed_text"] = df["incident_text"].apply(preprocess_text)
    
    X = df["processed_text"]
    y_type = df["incident_type"]
    y_severity = df["severity"]
    
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)
    
    # Save vectorizer
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    
    # --- Incident Type Classification ---
    print("\n--- Training Incident Type Classifier ---")
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y_type, test_size=0.2, random_state=42, stratify=y_type)
    
    # Use LightGBM for speed and handling imbalance
    clf_type = lgb.LGBMClassifier(class_weight="balanced", random_state=42)
    clf_type.fit(X_train, y_train)
    
    y_pred_type = clf_type.predict(X_test)
    print("Incident Type Classification Report:")
    print(classification_report(y_test, y_pred_type))
    
    joblib.dump(clf_type, os.path.join(model_dir, "type_classifier.pkl"))
    
    # --- Severity Classification ---
    print("\n--- Training Severity Classifier ---")
    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(X_vec, y_severity, test_size=0.2, random_state=42, stratify=y_severity)
    
    clf_sev = lgb.LGBMClassifier(class_weight="balanced", random_state=42)
    clf_sev.fit(X_train_sev, y_train_sev)
    
    y_pred_sev = clf_sev.predict(X_test_sev)
    print("Severity Classification Report:")
    print(classification_report(y_test_sev, y_pred_sev))
    
    joblib.dump(clf_sev, os.path.join(model_dir, "severity_classifier.pkl"))
    
    print(f"\nModels saved successfully in {model_dir}")

if __name__ == "__main__":
    train_and_evaluate_models()
