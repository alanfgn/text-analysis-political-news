import pandas as pd
from enum import Enum
import os

base_lexicon_path =  os.path.join("datasets") 

class LetterTypes(Enum):
    Adj = "adj"
    Nom = "nom"
    Verb = "verb"
    Idiom = "idiom"
    Det = "det"
    Emot = "emot"
    Htag = "htag"

class Lexicons():

    sentilex_type = {
        "Adj": LetterTypes.Adj,
        "N": LetterTypes.Nom,
        "V": LetterTypes.Verb,
        "IDIOM": LetterTypes.Idiom
    }

    oplexicon_type = {
        'adj': LetterTypes.Adj,
        'n': LetterTypes.Nom,
        'vb': LetterTypes.Verb,
        'det': LetterTypes.Det,
        'emot': LetterTypes.Emot,
        'htag': LetterTypes.Htag
    }

    sentilex = None
    oplexicon = None

    @staticmethod
    def _get_pandas_from_path(path):
        return pd.read_csv(path)

    @staticmethod
    def get_sentilex():
        if Lexicons.sentilex is None:
            Lexicons.sentilex = Lexicons._get_pandas_from_path(
                "%s/sentiLex_lem_PT02.csv" % base_lexicon_path)
        return Lexicons.sentilex

    @staticmethod
    def get_oplexicon():
        if Lexicons.oplexicon is None:
            Lexicons.oplexicon = Lexicons._get_pandas_from_path(
                "%s/oplexicon_v3.0.csv" % base_lexicon_path)
        return Lexicons.oplexicon


def diference():
    opl3 = Lexicons.get_oplexicon()
    sent = Lexicons.get_sentilex()

    opl3[[line[1] in sent.term.values and sent[sent['term'] == line[1]]
          ['polarity'].values[0] != line[3] for line in opl3.values]]

def polarity_oplexicon_document(document):
    opl3 = Lexicons.get_oplexicon()
    return sum(opl3[opl3['term'].isin(document)]['polarity'].values)

def categs_pol_oplexicon_document(document):
    opl3 = Lexicons.get_oplexicon()
    categs_pols = []

    for token in document:
        arr = opl3.loc[opl3['term'] == token]['polarity'].values
        if(arr.size == 0):
            categs_pols.append(None)        
        else:
            categs_pols.append(arr[0])    

    return categs_pols

def polarity_sentilex_document(document):
    sent = Lexicons.get_sentilex()
    return sum(sent[sent['term'].isin(document)]['polarity'].values)

def categs_pol_sentilex_document(document):
    sent = Lexicons.get_sentilex()
    categs_pols = []

    for token in document:
        arr = sent.loc[sent['term'] == token]['polarity'].values
        if(arr.size == 0):
            categs_pols.append(None)        
        else:
            categs_pols.append(arr[0])    

    return categs_pols
