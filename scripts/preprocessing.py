import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = " ".join([w for w in text.split() if w not in stopwords.words("english")])
    stemmer = SnowballStemmer("english")
    return " ".join([stemmer.stem(w) for w in text.split()])

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df[["tweet_text", "cyberbullying_type"]].dropna()
    df["clean_text"] = df["tweet_text"].apply(clean_text)
    return df

def vectorize_text(corpus):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(corpus).toarray()
    return X, vectorizer
