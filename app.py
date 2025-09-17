# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Title and description
# -------------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("This app uses a simple Machine Learning model to classify news text as **FAKE** or **REAL**.")

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/fake.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()
if df is not None:
    st.subheader("📂 Dataset Preview")
    st.write(df.head())
else:
    st.stop()

# -------------------------------
# Prepare data
# -------------------------------
X = df["text"].astype(str)  # news articles
y = np.ones(len(X))         # since fake.csv only has FAKE, we treat it as 1

# For demo: create some dummy "real" labels (to balance training)
real_samples = X.sample(min(2000, len(X)), random_state=42).apply(lambda x: x.replace("fake", "real"))
X = pd.concat([X, real_samples])
y = np.concatenate([y, np.zeros(len(real_samples))])

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
        if prediction == 1:
            st.error("❌ This news is predicted as **FAKE**")
        else:
            st.success("✅ This news is predicted as **REAL**")
