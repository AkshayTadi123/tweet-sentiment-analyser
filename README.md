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

Cleaning drops empty, duplicate and conflicting-label tweets (14,640 → 14,402):

| Split | Negative | Neutral | Positive | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 6,355 | 2,132 | 1,594 | 10,081 |
| Val | 1,362 | 457 | 341 | 2,160 |
| Test | 1,362 | 457 | 342 | 2,161 |
| **All** | **9,079 (63%)** | **3,046 (21%)** | **2,277 (16%)** | **14,402** |

Exploration lives in `notebooks/01_eda.ipynb`.
