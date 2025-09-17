# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Title and description
# -------------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("This app classifies news articles as **FAKE** or **REAL** using Machine Learning.")

# -------------------------------
# Load datasets
# -------------------------------
@st.cache_data
def load_data():
    try:
        df_fake = pd.read_csv("data/fake.csv")
        df_true = pd.read_csv("data/true.csv")

        df_fake["label"] = 1   # 1 → Fake
        df_true["label"] = 0   # 0 → Real

        df = pd.concat([df_fake, df_true], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()
if df is not None:
    st.subheader("📂 Dataset Preview")
    st.write(df.head())
    st.write(f"Dataset size: {df.shape[0]} rows")
else:
    st.stop()

# -------------------------------
# Prepare data
# -------------------------------
X = df["text"].astype(str)  # news text
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
st.success(f"✅ Model trained with Accuracy: {acc*100:.2f}%")

with st.expander("See classification report"):
    st.text(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

# -------------------------------
# User Input for Prediction
# -------------------------------
st.subheader("🔎 Try it yourself")
user_input = st.text_area("Enter news text here:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        prob = model.predict_proba(input_tfidf)[0]

        if prediction == 1:
            st.error(f"❌ Predicted as **FAKE** (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.success(f"✅ Predicted as **REAL** (Confidence: {prob[0]*100:.2f}%)")
