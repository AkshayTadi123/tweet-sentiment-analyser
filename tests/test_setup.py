from nltk.corpus import stopwords, wordnet


def test_package_imports():
    import sentiment  # noqa: F401


def test_nltk_resources_available():
    assert "not" in stopwords.words("english")
    assert wordnet.synsets("happy")
