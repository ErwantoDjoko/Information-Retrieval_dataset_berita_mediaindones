import streamlit as st
from search_engine import load_articles, search

st.title("Pencarian Berita")

@st.cache_resource
def load_data():
    return load_articles("articles_last_7_days.json")

articles, vectorizer, vectors = load_data()

query = st.text_input("Cari berita:")

if query:
    results = search(query, articles, vectorizer, vectors)
    if results:
        for res in results:
            st.subheader(res["title"])
            st.write(f"📰 {res['domain']} | 🗓️ {res['publish_date']}")
            st.write(res["snippet"])
            st.write(f"🔍 Skor: {res['score']:.4f}")
            st.markdown("---")
    else:
        st.warning("Tidak ada hasil yang ditemukan.")