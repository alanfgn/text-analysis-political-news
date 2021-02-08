import unicodedata
import unidecode
import nltk
import spacy
import csv
from spacy import displacy
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize, MWETokenizer

# nltk.download('rslp')

CLEAN_UNICODE_CATEGORIES = ['Pc', 'Pd',
                            'Pe', 'Pf', 'Pi', 'Po', 'Ps', 'Sk', 'So', "Z", "Zp", "Zs", "Cc"]


class SpacyInstance():

    SPACY_NLP = None
    MERGE_PIPE = False

    @staticmethod
    def get_instance():
        if SpacyInstance.SPACY_NLP is None:
            SpacyInstance.SPACY_NLP = spacy.load("pt_core_news_lg")

        return SpacyInstance.SPACY_NLP

    @staticmethod
    def get_merge_ent_instance():
        if not SpacyInstance.MERGE_PIPE:
            nlp = SpacyInstance.get_instance()
            merge_ents = nlp.create_pipe("merge_entities")
            nlp.add_pipe(merge_ents)
            SpacyInstance.MERGE_PIPE = True

        return SpacyInstance.get_instance()

    @staticmethod
    def clean_instance():
        SpacyInstance.SPACY_NLP = None
        SpacyInstance.MERGE_PIPE = False

    @staticmethod
    def get_normal_instance():
        if SpacyInstance.MERGE_PIPE:
            SpacyInstance.clean_instance()

        return SpacyInstance.get_instance()

def paragraph_segmentation(text):
    return text.split("\n")


def sentence_segmentation(text):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    return sent_tokenizer.tokenize(text)


def word_segmentation(text):
    return word_tokenize(text, language="portuguese")


def tokenize_mwe(text):
    nlp = SpacyInstance.get_merge_ent_instance()
    return [token.text for token in nlp(text)]


def clean_tokenize_mwe(text):
    nlp = SpacyInstance.get_merge_ent_instance()
    return [token.text for token in nlp(text) if not token.is_stop]


def clean_tokens(words, stopwords=nltk.corpus.stopwords.words("portuguese")):
    for word in words:

        if word.lower() in stopwords:
            continue
        if all(unicodedata.category(char) in CLEAN_UNICODE_CATEGORIES for char in word):
            continue

        yield word.lower()


def remove_acentuation(document):
    for token in document:
        yield unidecode.unidecode(token)


def stemmer_doc(document):
    stemmer = nltk.stem.RSLPStemmer()
    for token in document:
        yield stemmer.stem(token)


def clean_corpus(corpus):
    for doc, fileid in corpus:
        yield (clean_tokens(doc), fileid)
