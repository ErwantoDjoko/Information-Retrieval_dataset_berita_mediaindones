import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
from nltk.tokenize import word_tokenize
import nltk
import re

nltk.download('punkt')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

class TopicIRModel:
    def __init__(self, docs):
        self.raw_docs = docs
        self.docs = [clean_text(doc) for doc in docs]
        self.tokens = [word_tokenize(doc) for doc in self.docs]

        # LDA
        self.dictionary = corpora.Dictionary(self.tokens)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.tokens]
        self.lda_model = models.LdaModel(self.corpus, num_topics=3, id2word=self.dictionary, passes=10)

        # TF-IDF
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs)

    def get_topics(self, num_words=5):
        return self.lda_model.print_topics(num_words=num_words)

    def search(self, query, top_n=3):
        query_clean = clean_text(query)
        query_vec = self.vectorizer.transform([query_clean])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        results = [(self.raw_docs[i], similarities[i]) for i in top_indices]
        return results
