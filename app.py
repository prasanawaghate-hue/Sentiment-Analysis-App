import streamlit as st
import pickle
import re

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below and I'll predict whether it's positive or negative.")

user_input = st.text_area("Your review:", height=150, placeholder="Type or paste a movie review here...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(user_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.success(f"✅ Positive Sentiment (confidence: {proba[1]*100:.1f}%)")
        else:
            st.error(f"❌ Negative Sentiment (confidence: {proba[0]*100:.1f}%)")

        st.write("Confidence breakdown:")
        st.progress(float(proba[1]))
        col1, col2 = st.columns(2)
        col1.metric("Negative", f"{proba[0]*100:.1f}%")
        col2.metric("Positive", f"{proba[1]*100:.1f}%")