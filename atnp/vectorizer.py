import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.text import TextCollection
from gensim.models import Word2Vec


def identity_tokenizer(text):
    return list(text)

def generate_tf_idf(corpus):
    tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    return tfidf.fit_transform(corpus), tfidf

def generate_count(corpus):
    count = CountVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    return count.fit_transform(corpus), count

def generate_word2vec(corpus):
    return Word2Vec(corpus, min_count=1, vector_size=100, workers=4, window=3, sg=1)


def load_word2vec(path):
    return Word2Vec.load(path)
