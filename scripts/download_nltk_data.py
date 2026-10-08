"""Download the NLTK resources the preprocessing step needs."""
import ssl
import sys

import certifi
import nltk

# TweetTokenizer needs no download; stopwords and WordNet (lemmatization) do.
RESOURCES = ["stopwords", "wordnet", "omw-1.4"]

if __name__ == "__main__":
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

    failed = [r for r in RESOURCES if not nltk.download(r, quiet=True)]
    if failed:
        sys.exit(f"Failed to download: {', '.join(failed)}")
    print(f"Downloaded: {', '.join(RESOURCES)}")
