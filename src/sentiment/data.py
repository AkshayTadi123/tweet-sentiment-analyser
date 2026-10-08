"""Load, clean and split the Twitter US Airline Sentiment dataset."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "Tweets.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
SEED = 42


def load_raw(path=RAW_PATH):
    return pd.read_csv(path)


def clean(df):
    """Keep the columns the pipeline uses; drop empty and duplicate tweets."""
    df = df.rename(columns={"airline_sentiment": "label", "tweet_created": "created_at"})
    df = df[["text", "label", "created_at", "airline"]].copy()
    df["text"] = df["text"].str.strip()
    df = df[df["text"].notna() & (df["text"] != "")]
    # The same text with different labels means annotators disagreed: drop every copy.
    conflicting = df.groupby("text")["label"].transform("nunique") > 1
    df = df[~conflicting].drop_duplicates(subset="text")
    df["created_at"] = pd.to_datetime(df["created_at"], format="%Y-%m-%d %H:%M:%S %z")
    return df.reset_index(drop=True)


def split(df):
    """Stratified 70/15/15 train/val/test split."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)
    return train, val, test


def main():
    df = clean(load_raw())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in zip(["train", "val", "test"], split(df)):
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows, {part['label'].value_counts().to_dict()}")
    print(f"total: {len(df)} rows, {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
