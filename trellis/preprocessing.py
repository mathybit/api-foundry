import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class InputPreprocessor:
    """Clean and normalize raw text for downstream TF-IDF + classification."""

    def __init__(self):
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words("english"))

    def preprocess(self, text: str) -> list[str]:
        """Return a list of cleaned, lemmatized tokens from raw text.

        Pipeline:
            1. Lowercase
            2. Tokenize (nltk.word_tokenize)
            3. Strip non-alpha tokens (punctuation, digits)
            4. Remove stopwords
            5. Lemmatize
            6. Drop residual tokens <= 1 char
        """
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha()]
        tokens = [t for t in tokens if t not in self._stop_words]
        tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 1]
        return tokens
