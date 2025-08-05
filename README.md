# Product Review Sentiment + Reasoning Tool (Offline)

This version runs entirely offline (no API keys, no Hugging Face).

## Features
- Sentiment classification (Positive/Negative) using Logistic Regression + TF-IDF
- Reasoning: keyword/phrase-based cues (offline)
- Rephrase: rule-based, brand-friendly softening (offline)

## How to Run
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python model.py
python -m streamlit run app.py
```


public URL:https://appuct-review-sentiment-offline-kaytkjfgqmr59twvkf2efi.streamlit.app
