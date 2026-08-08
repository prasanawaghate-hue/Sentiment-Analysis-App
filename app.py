import streamlit as st
import pickle
import re
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Sentiment AI | Movie Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

# Cache model loading for performance
@st.cache_resource
def load_assets():
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_assets()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Custom Styling
st.markdown("""
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .main-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #3a3a55;
    }
    .main-header h1 { color: #ffffff; margin: 0; font-weight: 700; font-size: 2.2rem; }
    .main-header p { color: #a0a0b8; margin-top: 0.5rem; font-size: 1rem; }
    div.stButton > button:first-child {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white; font-weight: 600; font-size: 1.05rem; border: none;
        padding: 0.65rem 1rem; border-radius: 8px; transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #e03e3e 0%, #f05555 100%);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown("""
    <div class="main-header">
        <h1>🎬 Movie Review Sentiment AI</h1>
        <p>Analyze film review emotion instantly using NLP & Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Mode Selection")
    mode = st.radio("Choose Analysis Mode:", ["Single Review Mode", "Bulk Analysis Mode (Multiple)"])
    st.divider()
    st.header("ℹ️ App Overview")
    st.write("Powered by TF-IDF Vectorization & Logistic Regression trained on 50,000 IMDB movie reviews.")

# ================= MODE 1: SINGLE REVIEW =================
if mode == "Single Review Mode":
    st.markdown("### ✍️ Enter Single Review")
    
    sample_choice = st.selectbox(
        "Or pick a sample review:",
        [
            "Select a sample...",
            "Positive: Absolutely stunning visual effects and phenomenal performances! A true masterpiece.",
            "Negative: The plot was predictable, pacing was horribly slow, and I wanted to leave mid-way."
        ]
    )
    
    default_text = "" if sample_choice == "Select a sample..." else sample_choice[10:]
    
    user_input = st.text_area("Review Content", value=default_text, height=140, placeholder="Type or paste a movie review here...", label_visibility="collapsed")
    analyze_btn = st.button("✨ Analyze Sentiment")

    if analyze_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter a review first.")
        else:
            cleaned = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            
            st.divider()
            st.markdown("### 📊 Analysis Results")
            if prediction == 1:
                st.success(f"### 🎉 Positive Sentiment\n**Confidence:** {proba[1]*100:.1f}%")
            else:
                st.error(f"### 👎 Negative Sentiment\n**Confidence:** {proba[0]*100:.1f}%")

            col1, col2 = st.columns(2)
            col1.metric("Negative Score", f"{proba[0]*100:.1f}%")
            col2.metric("Positive Score", f"{proba[1]*100:.1f}%")
            st.progress(float(proba[1]))

# ================= MODE 2: BULK ANALYSIS MODE =================
else:
    st.markdown("### 📝 Bulk Review Batch Analysis")
    st.write("Paste multiple reviews below. **Separate each review with a new line (Enter key).**")

    sample_bulk = (
        "This film was an absolute masterpiece with incredible acting!\n"
        "Boring plot, terrible pacing, and a complete waste of time.\n"
        "Decent watch with good visuals, though the ending felt rushed.\n"
        "One of the worst movies I have ever seen in my life.\n"
        "Brilliant direction and stellar performances all around!"
    )

    bulk_input = st.text_area(
        "Bulk Reviews",
        value=sample_bulk,
        height=200,
        placeholder="Paste multiple reviews here (one per line)...",
        label_visibility="collapsed"
    )

    bulk_btn = st.button("🚀 Process Batch Analysis")

    if bulk_btn:
        reviews = [r.strip() for r in bulk_input.split('\n') if r.strip()]
        
        if not reviews:
            st.warning("⚠️ Please enter at least one review.")
        else:
            cleaned_reviews = [clean_text(r) for r in reviews]
            vectorized_reviews = tfidf.transform(cleaned_reviews)
            predictions = model.predict(vectorized_reviews)
            probabilities = model.predict_proba(vectorized_reviews)

            pos_count = sum(predictions == 1)
            neg_count = sum(predictions == 0)
            total = len(reviews)

            st.divider()
            st.markdown("### 📈 Batch Overview")

            # Metric Summary Cards
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reviews Evaluated", total)
            col2.metric("Positive Reviews", f"{pos_count} ({pos_count/total*100:.0f}%)")
            col3.metric("Negative Reviews", f"{neg_count} ({neg_count/total*100:.0f}%)")

            # Chart Visual
            chart_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Count": [pos_count, neg_count]
            })
            st.bar_chart(chart_data, x="Sentiment", y="Count")

            # Itemized Results Table
            st.markdown("### 📋 Detailed Itemized Breakdown")
            results_df = pd.DataFrame({
                "Review Text": reviews,
                "Predicted Sentiment": ["Positive 🎉" if p == 1 else "Negative 👎" for p in predictions],
                "Positive Probability": [f"{prob[1]*100:.1f}%" for prob in probabilities]
            })
            st.dataframe(results_df, use_container_width=True)