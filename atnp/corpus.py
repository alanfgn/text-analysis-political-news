import pandas as pd
import pickle
import codecs
import json
import os
from nltk.corpus.reader.api import CorpusReader, CategorizedCorpusReader

TEXT_PATTERN = r'[\w0-9-#_\.#+,]+\.txt'
PKL_PATTERN = r'[\w0-9-#_\.#+,]+\.pickle'
CAT_FILE = "cat.csv"


class RawCorpusReader(CorpusReader):
    def __init__(
            self,
            root,
            fileids,
            encoding='utf8',
            **kwargs):

        CorpusReader.__init__(self, root, fileids, encoding)

    def raws(self, fileids=None):
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield json.load(f)


class CategoriesReader():

    def __init__(self, root, file_path, header=None):
        self.path = os.path.join(root, file_path)
        self.header = header
        self.csv = None

    def __init_df(self):
        if self.csv is None:
            self.csv = pd.read_csv(self.path, delimiter=" ", names=self.header)

    def get(self, fileid, cat):
        self.__init_df()
        return self.csv[self.csv['fileid'] == fileid][cat].values[0]

    def get_cats(self, cat):
        self.__init_df()
        return list(pd.unique(self.csv[cat]))


class TokenizedCorpusReader(CategorizedCorpusReader, CorpusReader):

    def __init__(self, root, fileids=TEXT_PATTERN, cat_file=CAT_FILE, header=None, **kwargs):
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_file'] = CAT_FILE

        self.categ = CategoriesReader(root, cat_file, header)

        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def resolve(self, fileids, categories):
        if categories is not None:
            return self.fileids(categories)

        return fileids

    def documents(self, fileids=None, categories=None):
        fileids = self.resolve(fileids, categories)

        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'r') as f:
                yield f.read(), fileid

    def apply(self, functions, fileids=None, categories=None):
        for document, fileid in self.documents(fileids, categories):
            yield self.__get_tokens__(document, functions), fileid

    def __get_tokens__(self, document, functions):
        def extract(doc, funcs):
            if len(funcs) == 1:
                return funcs[0](doc)
            else:
                return extract(funcs[0](doc), funcs[1:])

        return extract(document, functions)


class PickledCorpusReader(CategorizedCorpusReader, CorpusReader):

    def __init__(self, root, fileids=PKL_PATTERN, cat_file=CAT_FILE, header=None, **kwargs):
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_file'] = CAT_FILE

        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def resolve(self, fileids, categories):
        if categories is not None:
            return self.fileids(categories)

        return fileids

    def pickles(self, fileids=None, categories=None):
        fileids = self.resolve(fileids, categories)

        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

