import pandas as pd

from sentiment.data import clean, split


def raw(rows):
    return pd.DataFrame(rows, columns=["airline_sentiment", "text", "tweet_created", "airline", "extra"])


def test_clean_keeps_and_renames_columns():
    df = clean(raw([["positive", "great flight", "2015-02-24 11:35:52 -0800", "Delta", 1]]))
    assert list(df.columns) == ["text", "label", "created_at", "airline"]
    assert df.loc[0, "created_at"].year == 2015


def test_clean_drops_empty_duplicate_and_conflicting_text():
    t = "2015-02-24 11:35:52 -0800"
    df = clean(raw([
        ["positive", "great flight", t, "Delta", 0],
        ["positive", "great flight ", t, "Delta", 0],  # duplicate after strip
        ["negative", "   ", t, "Delta", 0],             # empty
        ["negative", "ok I guess", t, "Delta", 0],      # conflicting labels
        ["neutral", "ok I guess", t, "Delta", 0],
    ]))
    assert df["text"].tolist() == ["great flight"]


def test_split_is_stratified_and_disjoint():
    labels = ["negative"] * 60 + ["neutral"] * 20 + ["positive"] * 20
    df = pd.DataFrame({"text": [f"tweet {i}" for i in range(100)], "label": labels})
    train, val, test = split(df)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert set(train.index).isdisjoint(val.index) and set(val.index).isdisjoint(test.index)
    assert (train["label"] == "negative").mean() == 0.6
