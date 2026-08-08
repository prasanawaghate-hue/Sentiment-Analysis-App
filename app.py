import streamlit as st
import pickle
import re

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

# Custom CSS Injection
st.markdown("""
    <style>
    /* Center and constrain main content width */
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Header Card Banner */
    .main-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #3a3a55;
    }
    
    .main-header h1 {
        color: #ffffff;
        margin: 0;
        font-weight: 700;
        font-size: 2.2rem;
    }
    
    .main-header p {
        color: #a0a0b8;
        margin-top: 0.5rem;
        font-size: 1rem;
    }

    /* Gradient Call-To-Action Button */
    div.stButton > button:first-child {
        width: 100%;
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        border: none;
        padding: 0.65rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
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

# Sidebar Context
with st.sidebar:
    st.header("ℹ️ App Overview")
    st.write("This app uses TF-IDF Vectorization paired with Logistic Regression trained on 50,000 IMDB movie reviews.")
    st.divider()
    st.subheader("💡 Best Usage")
    st.markdown("""
    - Paste multi-sentence reviews.
    - Test explicit positive/negative words.
    - Evaluates confidence probability percentages.
    """)

# Main Content Card
with st.container():
    st.markdown("### ✍️ Enter Review")
    
    # Pre-built sample picker for quick live demos
    sample_choice = st.selectbox(
        "Or pick a sample review to test:",
        [
            "Select a sample...",
            "Positive: Absolutely stunning visual effects and phenomenal performances! A true masterpiece.",
            "Negative: The plot was predictable, pacing was horribly slow, and I wanted to leave mid-way."
        ]
    )
    
    default_text = "" if sample_choice == "Select a sample..." else sample_choice[10:]
    
    user_input = st.text_area(
        "Review Content",
        value=default_text,
        height=140,
        placeholder="Type or paste a movie review here...",
        label_visibility="collapsed"
    )

    analyze_btn = st.button("✨ Analyze Sentiment")

# Results Output Section
if analyze_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter a review or select a sample above.")
    else:
        with st.spinner("Processing text..."):
            cleaned = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            proba = model.predict_proba(vectorized)[0]
            
            neg_score = proba[0] * 100
            pos_score = proba[1] * 100

        st.divider()
        st.markdown("### 📊 Analysis Results")

        if prediction == 1:
            st.success(f"### 🎉 Positive Sentiment\n**Confidence:** {pos_score:.1f}%")
        else:
            st.error(f"### 👎 Negative Sentiment\n**Confidence:** {neg_score:.1f}%")

        # Metrics Breakdown
        col1, col2 = st.columns(2)
        col1.metric("Negative Score", f"{neg_score:.1f}%")
        col2.metric("Positive Score", f"{pos_score:.1f}%")

        # Visual Confidence Scale
        st.write("**Positivity Scale:**")
        st.progress(float(proba[1]))