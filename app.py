
import streamlit as st
#from flask import Flask, request, render_template
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

app = Flask(__name__)
stop_words = set(stopwords.words('indonesian'))

# --- Preprocessing ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# # --- Load Data ---
# def load_articles(json_file):
#     with open(json_file, 'r', encoding='utf-8') as f:
#         raw_data = json.load(f)

#     articles = []
#     for domain, items in raw_data.items():
#         for item in items:
#             text = item.get("text", "")
#             if text.strip():
#                 articles.append({
#                     "domain": domain,
#                     "title": item.get("title", ""),
#                     "publish_date": item.get("publish_date", ""),
#                     "text": text
#                 })
#     return articles

def load_articles(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        print("📄 Tipe data JSON:", type(raw_data))  # Debug: periksa tipe data
        if isinstance(raw_data, list):
            print("🔍 Contoh isi JSON (list):", raw_data[:1])
        elif isinstance(raw_data, dict):
            print("🔍 Contoh isi JSON (dict):", list(raw_data.items())[:1])

    articles = []

    # Pengecekan apakah raw_data adalah dict
    if isinstance(raw_data, dict):
        for item in raw_data.items():  # Mengambil pasangan key dan value
            if isinstance(item, tuple) and len(item) == 2:  # Pastikan item adalah tuple dengan dua elemen
                domain, items = item
                if isinstance(items, list):  # Pastikan items adalah list
                    for article in items:
                        if isinstance(article, dict) and "text" in article:  # Pastikan article memiliki key 'text'
                            articles.append({
                                "domain": domain,
                                "title": article.get("title", ""),
                                "publish_date": article.get("publish_date", ""),
                                "text": article.get("text", "")
                            })
            else:
                print("⚠️ Format item tidak sesuai:", item)  # Debug jika format tidak sesuai

    # Jika raw_data adalah list (tidak ada domain)
    elif isinstance(raw_data, list):
        for item in raw_data:
            if isinstance(item, dict) and "text" in item:
                articles.append({
                    "domain": item.get("domain", "unknown"),  # Jika tidak ada domain, berikan 'unknown'
                    "title": item.get("title", ""),
                    "publish_date": item.get("publish_date", ""),
                    "text": item.get("text", "")
                })

    else:
        raise ValueError("Format JSON tidak dikenali: harus dict atau list.")

    return articles



# --- TF-IDF Vectorization ---
articles = load_articles('articles_last_7_days.json')
corpus = [preprocess(a['text']) for a in articles]
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(corpus)

# --- Search Function ---
def search(query, top_n=5):
    query_processed = preprocess(query)
    query_vec = vectorizer.transform([query_processed])
    scores = cosine_similarity(query_vec, vectors).flatten()

    print("Skor cosine similarity untuk query '{}':".format(query))
    print(scores)  # Debugging untuk melihat skor similarity
    
    ranked_indices = scores.argsort()[::-1][:top_n]
    results = []
    
    # Filter dan pastikan artikel relevan dengan query
    for idx in ranked_indices:
        title = articles[idx]['title']
        domain = articles[idx]['domain']
        publish_date = articles[idx]['publish_date']
        text = articles[idx]['text']
        
        # Cek apakah kata yang relevan dengan query muncul dalam artikel
        if query.lower() in title.lower() or query.lower() in text.lower():
            results.append({
                "title": title,
                "domain": domain,
                "publish_date": publish_date,
                "score": float(scores[idx]),
                "snippet": text[:300]
            })
    
    return results


# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ''
    if request.method == 'POST':
        query = request.form['query']
        results = search(query, top_n=5)
    return render_template('index.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)

