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

## Data

Download [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) from Kaggle and save `Tweets.csv` to `data/raw/Tweets.csv`. Then build the stratified 70/15/15 train/val/test split:

```bash
python -m sentiment.data
```

Exploration lives in `notebooks/01_eda.ipynb`.
