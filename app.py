import streamlit as st
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Unduh resource NLTK ---
nltk.download('punkt')
nltk.download('stopwords')

# --- Preprocessing ---
stop_words = set(stopwords.words('indonesian'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# --- Load Data ---
@st.cache_data
def load_articles(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    articles = []

    if isinstance(raw_data, dict):
        for domain, items in raw_data.items():
            if isinstance(items, list):
                for article in items:
                    if isinstance(article, dict) and "text" in article:
                        articles.append({
                            "domain": domain,
                            "title": article.get("title", ""),
                            "publish_date": article.get("publish_date", ""),
                            "text": article.get("text", "")
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
    else:
        raise ValueError("Format JSON tidak dikenali: harus dict atau list.")

    return articles

# Load articles
articles = load_articles("articles_last_7_days.json")
corpus = [preprocess(a['text']) for a in articles]
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(corpus)

# --- Search Function ---
def search(query, top_n=5):
    query_processed = preprocess(query)
    query_vec = vectorizer.transform([query_processed])
    scores = cosine_similarity(query_vec, vectors).flatten()
    ranked_indices = scores.argsort()[::-1][:top_n]

    results = []
    for idx in ranked_indices:
        title = articles[idx]['title']
        domain = articles[idx]['domain']
        publish_date = articles[idx]['publish_date']
        text = articles[idx]['text']
        if query.lower() in title.lower() or query.lower() in text.lower():
            results.append({
                "title": title,
                "domain": domain,
                "publish_date": publish_date,
                "score": float(scores[idx]),
                "snippet": text[:300] + "..."
            })
    return results

# --- Streamlit UI ---
st.title("Sistem Pencarian Berita")

query = st.text_input("Masukkan kata kunci untuk mencari berita:")
if st.button("Cari"):
    if query.strip() == "":
        st.warning("Silakan masukkan kata kunci terlebih dahulu.")
    else:
        results = search(query)
        if results:
            for result in results:
                st.write(f"### {result['title']}")
                st.write(f"📅 {result['publish_date']} | 🌐 {result['domain']}")
                st.write(result['snippet'])
                st.markdown("---")
        else:
            st.warning("Tidak ada hasil ditemukan untuk kata kunci tersebut.")
