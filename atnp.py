from atnp import collecter, parser
from atnp import analysis

path_urls = "./data/Urls.csv"
path_raw = "./data/raw/"
path_ignore = "./data/Ignore.csv"
path_corpus = "./data/corpus/"
path_pickle = "./data/pickle/"
path_docs = "./docs/"


labels_title = {
    "impeachment-dilma": "Impeachment de Dilma Rousseff",
    "reforma-trabalhista": "Reforma Trabalhista",
    "afastamento-aecio": "Afastamento de Aécio Neves",
    "prisao-lula": "Prisão de Lula",
    "soltura-lula": "Soltura de Lula",
    "reforma-previdencia": "Reforma da Previdência",
    "subject": "Tema",
    "journal": "Veiculo"
}


def collect_phase():
    collecter.download(path_urls, path_raw)
    collecter.report(path_urls, path_raw)


def parse_phase():
    parser.transform(path_raw, path_corpus, path_ignore)


def analysis_phase():
    return analysis.Analyser(path_corpus, path_pickle, path_docs, labels_title)


an = analysis_phase()
