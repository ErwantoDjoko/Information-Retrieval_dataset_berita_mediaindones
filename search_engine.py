import json
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('indonesian'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

def load_articles(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    articles = []
    if isinstance(raw_data, dict):
        for domain, items in raw_data.items():
            for item in items:
                if isinstance(item, dict) and "text" in item:
                    articles.append({
                        "domain": domain,
                        "title": item.get("title", ""),
                        "publish_date": item.get("publish_date", ""),
                        "text": item.get("text", "")
                    })
    elif isinstance(raw_data, list):
        for item in raw_data:
            if isinstance(item, dict) and "text" in item:
                articles.append({
                    "domain": item.get("domain", "unknown"),
                    "title": item.get("title", ""),
                    "publish_date": item.get("publish_date", ""),
                    "text": item.get("text", "")
                })
    corpus = [preprocess(a['text']) for a in articles]
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    vectors = vectorizer.fit_transform(corpus)
    return articles, vectorizer, vectors

def search(query, articles, vectorizer, vectors, top_n=5):
    query_processed = preprocess(query)
    query_vec = vectorizer.transform([query_processed])
    scores = cosine_similarity(query_vec, vectors).flatten()
    ranked_indices = scores.argsort()[::-1][:top_n]
    results = []
    for idx in ranked_indices:
        article = articles[idx]
        results.append({
            "title": article["title"],
            "domain": article["domain"],
            "publish_date": article["publish_date"],
            "score": float(scores[idx]),
            "snippet": article["text"][:300]
        })
    return results