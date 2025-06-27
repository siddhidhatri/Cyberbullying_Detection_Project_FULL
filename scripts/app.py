import streamlit as st
import joblib
from preprocessing import clean_text

# Load model and vectorizer
model = joblib.load("models/multiclass_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.set_page_config(page_title="Cyberbullying Detector", layout="centered")
st.title("🚨 Cyberbullying Detection & Type Classification")

st.markdown("Enter a social media comment below. The app will detect if it's **cyberbullying** and identify its **type**.")

# User input
user_input = st.text_area("💬 Enter a comment:")

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Clean and transform input
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]

        # Show result
        if prediction.lower() == "not_cyberbullying":
            st.success("✅ This comment is **NOT cyberbullying**.")
        else:
            st.error(f"🚫 This comment is **cyberbullying**.\n\n🔎 Type: **{prediction.upper()}**")
