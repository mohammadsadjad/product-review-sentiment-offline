import os
import json
import streamlit as st
import joblib
import numpy as np
from reasoning import explain_reason, rephrase_review

# Optional LLM (only if installed + key provided). Off by default.
USE_LLM = st.sidebar.checkbox("Use LLM (if available)", value=False)
openai_available = False
if USE_LLM:
    try:
        from openai import OpenAI
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        if OPENAI_API_KEY:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            openai_available = True
        else:
            st.sidebar.warning("OPENAI_API_KEY not set. Staying offline.")
    except Exception:
        st.sidebar.warning("OpenAI not installed. Staying offline.")

def explain_reason_llm(review: str, sentiment: str) -> str:
    try:
        model = "gpt-4o-mini"
        if sentiment == "Negative":
            prompt = f"Explain briefly why this review is negative:\n\n{review}"
        elif sentiment == "Positive":
            prompt = f"Explain briefly why this review is positive:\n\n{review}"
        else:
            prompt = f"Explain briefly why this review seems neutral:\n\n{review}"
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM unavailable) Offline explanation used. Reason: {e}"

def rephrase_review_llm(review: str) -> str:
    try:
        model = "gpt-4o-mini"
        prompt = (
            "Rephrase the following review in a neutral, polite, brand-friendly tone. "
            "Keep it concise and preserve the facts:\n\n" + review
        )
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM unavailable) Offline rephrase used. Reason: {e}"

st.set_page_config(page_title="Product Review Sentiment + Reasoning", page_icon="🛒")
st.title("🛒 Product Review Sentiment + Reasoning Tool")

# Load model & vectorizer
try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"Failed to load model/vectorizer. Did you run `python model.py`? Error: {e}")
    st.stop()

st.sidebar.header("Settings")
neutral_band = st.sidebar.slider("Neutral band (|P(Positive)−0.5| < x → Neutral)",
                                 min_value=0.02, max_value=0.20, value=0.08, step=0.01)
show_probs = st.sidebar.checkbox("Show prediction probabilities", value=True)

review = st.text_area("Enter a customer review:", height=150, placeholder="Paste a customer review here...")

def sentiment_with_neutral(prob_positive: float, band: float):
    delta = abs(prob_positive - 0.5)
    if delta < band:
        return "Neutral"
    return "Positive" if prob_positive >= 0.5 else "Negative"

def badge(text: str):
    color = {"Positive":"#16a34a","Neutral":"#64748b","Negative":"#dc2626"}.get(text, "#64748b")
    st.markdown(
        f"<span style='background:{color};color:white;padding:4px 10px;border-radius:999px;font-weight:600'>{text}</span>",
        unsafe_allow_html=True
    )

if st.button("Analyze"):
    if review.strip():
        review_vec = vectorizer.transform([review])

        # Proba and neutral logic
        try:
            proba = model.predict_proba(review_vec)[0]
            classes = list(model.classes_)           # ['Negative','Positive']
            idx_pos = classes.index("Positive")
            p_pos = float(proba[idx_pos])
        except Exception:
            p_pos = 1/(1+np.exp(-float(model.decision_function(review_vec))))
            proba = None

        sentiment = sentiment_with_neutral(p_pos, neutral_band)

        st.subheader("Result")
        cols = st.columns([1,4])
        with cols[0]:
            badge(sentiment)
        with cols[1]:
            if show_probs and proba is not None:
                st.caption(f"P(Positive) = {p_pos:.3f} | P(Negative) = {1-p_pos:.3f}")

        # Explanation
        if USE_LLM and openai_available:
            explanation = explain_reason_llm(review, sentiment)
            if explanation.startswith("(LLM unavailable)"):
                explanation = explain_reason(review, sentiment)
        else:
            explanation = explain_reason(review, sentiment)

        st.markdown("**Explanation:**")
        st.write(explanation)

        # Rephrase (editable)
        st.markdown("**Rephrase (editable):**")
        if USE_LLM and openai_available:
            rephrased_default = rephrase_review_llm(review)
            if rephrased_default.startswith("(LLM unavailable)"):
                rephrased_default = rephrase_review(review)
        else:
            rephrased_default = rephrase_review(review)

        rephrased = st.text_area("", value=rephrased_default, height=120)

        # Export
        result = {
            "review": review,
            "sentiment": sentiment,
            "p_positive": round(p_pos, 4),
            "explanation": explanation,
            "rephrased": rephrased
        }
        st.download_button(
            "Download Analysis (.json)",
            data=json.dumps(result, indent=2),
            file_name="review_analysis.json",
            mime="application/json"
        )

        # Copy to clipboard (client-side)
        st.markdown(
            '''
            <script>
            function copyRephrased() {
              const txt = document.getElementById('rephrase-box').value;
              navigator.clipboard.writeText(txt).then(()=>{alert('Rephrased text copied!');});
            }
            </script>
            <textarea id="rephrase-box" style="width:100%;height:0px;opacity:0;position:absolute;left:-9999px;">''' + 
            (rephrased_default if 'rephrased_default' in locals() else '') +
            '''</textarea>
            <button onClick="copyRephrased()">Copy Rephrased to Clipboard</button>
            ''',
            unsafe_allow_html=True
        )

    else:
        st.warning("Please enter a review first.")
