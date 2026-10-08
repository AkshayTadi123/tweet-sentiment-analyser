# Tweet Sentiment Analyser

Classifies tweets as positive, negative or neutral using TF-IDF features and logistic regression.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_nltk_data.py
pytest
```
