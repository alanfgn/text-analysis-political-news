# Text Analysis in Political News

This is a repository of a research work on Brazilian political news. 

> The article related to this research will be posted here soon

### Environment

All packages are in `requirements.txt.` 

Execute to create the environment in python.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
#### Datasets

This research used two datasets for sentimental analysis in Brazilian Portuguese . 

- [OpLexicon V3.0](https://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/oplexicon/)
- [SentiLex-PT 02](http://b2find.eudat.eu/dataset/b6bd16c2-a8ab-598f-be41-1e7aeecd60d3)

#### Models

This research used a model for SpaCy for NER and POS tagging.
- [SpaCy pt_core_news_lg-2.3.0](https://github.com/explosion/spacy-models/releases//tag/pt_core_news_lg-2.3.0)

### Collecting

>  Collecting raw corpus.

Based on handmade labeled dataset [Urls](https://docs.google.com/spreadsheets/d/1DKh_vWKLN4PwLARaE-wSwiDSSLSiTq45biTInK6vDn0/edit#gid=0) using  [Requests](https://requests.readthedocs.io/en/master/) to making requests.

```python
from atnp import collect

path_urls = "./data/Urls-Dados.csv"
path_raw = "./data/raw/"

collect.download(path_urls, path_raw)
collect.report(path_urls, path_raw)
```

Both methods are implemented with native python and [Requests](https://requests.readthedocs.io/en/master/) framework.

```python
# ...
request = requests.get(url)
_, domain, path = slice_url(url)

print("[%d] %s" % (request.status_code, url))

return {
    "fileid": gen_unique_name(domain, path),
    "url": url,
    "subject": row[header.index("subject")],
    "journal": row[header.index("journal")],
    "html": request.text
}

# ...
```

All functions is in the `collecter.py`

### Parsing

>  Making the corpus

Using [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) and [Readability](https://github.com/andreasvc/readability/) to clean HTML's with corpus reader of [Nltk](https://www.nltk.org/) with Ignored files dataset handmade pos individual analysis. 

```python
from atnp import parse

path_raw = "./data/raw/"
path_ignore = "./data/Ignore.csv"
path_corpus = "./data/corpus/"

parse.transform(path_raw, path_corpus, path_ignore)
```

Bellow a HTML parser method

```python
def html_parser(html):
    html = Document(html).summary()
    soup = bs4.BeautifulSoup(html, 'lxml')

    for element in soup.find_all(REMOVE_TAGS):
        element.extract()

    for element in soup.find_all("div", {"class": CONTENT_CLASSES}):
        element.extract()

    return "\n\n".join([
        element.text for element in soup.find_all(TEXT_TAGS)
        if not is_dispensable(element) and not is_not_impotant_text(element.text)
    ])

```

All functions is in the `parser.py`

### Preprocessing

> Tokenizing and normalize corpus functions.

In this stage both [NLTK](https://www.nltk.org/) and [SpaCy](https://spacy.io/) Model was used to tokenizer, steamming, and merge entities.

```python
def word_segmentation(text):
    return word_tokenize(text, language="portuguese")
# ...
def stemmer_doc(document):
    stemmer = nltk.stem.RSLPStemmer()
    for token in document:
        yield stemmer.stem(token)
# ...
def clean_tokenize_mwe(text):
    nlp = SpacyInstance.get_merge_ent_instance()
    return [token.text for token in nlp(text) if not token.is_stop]
```

All functions is in the `preproccessor.py`

### Vectorizing

> Vectorizing functions

Using [Gensim](https://radimrehurek.com/gensim/) and [Nltk](https://www.nltk.org/) to vectorizing the corpus.

```python
def generate_tf_idf(corpus):
    tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    return tfidf.fit_transform(corpus), tfidf

def generate_count(corpus):
    count = CountVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    return count.fit_transform(corpus), count

def generate_word2vec(corpus):
    return Word2Vec(corpus, min_count=1, vector_size=100, workers=4, window=3, sg=1)
```
All functions is in the `preproccessor.py`

### Clustering

> Making Clustering models 

Using [Scikit-learn](https://scikit-learn.org/stable/) to make the clustering models. 

- K-Means
- Mini Batch K-Means
- Mean Shift
- Affinity Propagation
- Agglomerative clustering 

```python
def generate_kmeans(corpus, clusters):
    model = KMeans(n_clusters=clusters, max_iter=500)
    model.fit_transform(corpus)
    return model
# ...
def generate_affinity_propagation(corpus):
    af = AffinityPropagation(max_iter=800, random_state=0, convergence_iter=30, verbose=True)
    af.fit(corpus)
    return af
# ...
def generate_agglomerative(corpus, linkage):
    ag = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=linkage)
    ag.fit(corpus)
    return ag
```

All functions is in the `clusters.py`

### Categorization

> Making Categorization models 

Using [Scikit-learn](https://scikit-learn.org/stable/) and [Keras](https://keras.io/) to make clustering models.

- Naive Bayes
- Svm
- Decision Trees
- Neural Network

```python
def generate_gaussian_nb(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    return gnb, (y_test, y_pred)
# ...
def generate_svm(x, y, kernel="precomputed"):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    clf = svm.SVC(kernel=kernel)
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    return clf, (y_test, y_pred)
# ...
def generate_nn(x, y, num_layers=100, activation='relu', loss='binary_crossentropy', optimizer='adam', **kwargs):
    le = LabelEncoder()
    le.fit(list(set(y)))

    y = tf.keras.utils.to_categorical(le.transform(y), len(le.classes_))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)

    model = Sequential()
    model.add(layers.Dense(num_layers, input_dim=x.shape[1], activation=activation))  
    # [...]
    model.add(layers.Dense(len(le.classes_), activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model
```

All functions is in the `categorization.py`

### Topic Analysis

> Making Topic Analysis models 

Using [Gensim](https://radimrehurek.com/gensim/) to making topic analysis model.

- LDA
- LSI

```python
def generate_lsi_model(corpus, num_topics=5):
    dictionary = gensim.corpora.Dictionary(corpus)
    
    dictionary.filter_extremes(no_below=4, no_above=0.6)
    transformed_corpus = [dictionary.doc2bow(text) for text in corpus]
    tfidf = gensim.models.TfidfModel(transformed_corpus)
    transformed_corpus = tfidf[transformed_corpus]

    return gensim.models.LsiModel(
        corpus=transformed_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        onepass=True,
        chunksize=500,
        power_iters=1000), dictionary, transformed_corpus

def generate_lda_model(corpus, num_topics=5):
    dictionary = gensim.corpora.Dictionary(corpus)

    dictionary.filter_extremes(no_below=4, no_above=0.6)
    transformed_corpus = [dictionary.doc2bow(text) for text in corpus]
    tfidf = gensim.models.TfidfModel(transformed_corpus)
    transformed_corpus = tfidf[transformed_corpus]

    return gensim.models.LdaModel(
        corpus=transformed_corpus,
        id2word=dictionary,
        chunksize=1740,
        alpha='auto',
        eta='auto',
        random_state=42,
        iterations=800,
        num_topics=num_topics,
        passes=20,
        eval_every=None), dictionary, transformed_corpus
```

All functions is in the `topics.py`

### Sentiment Analysis

> Making Sentiment Analysis 

Making Sentiment analysis with [OpLexicon V3.0](https://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/oplexicon/) and [SentiLex-PT 02](http://b2find.eudat.eu/dataset/b6bd16c2-a8ab-598f-be41-1e7aeecd60d3)

```python
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
```

All functions is in the `sentiment.py`

### Data visualization

> Visualizing models

Using [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), and [WordCloud](http://amueller.github.io/word_cloud/) to visualize with support of [Pandas](https://pandas.pydata.org/) and [Numpy](https://numpy.org/) to organize data and reductions functions provided by [Scikit-learn](https://scikit-learn.org/stable/).

```python
def scatter_plot_2d(x, labels):
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 7)

    df = pd.DataFrame(x, columns=['x', 'y'])
    df['label'] = labels

    for idx, label in enumerate(set(labels)):
        ax.scatter(df[df['label'] == label]["x"],
                   df[df['label'] == label]["y"],
                   label=label,
                   s=30, lw=0, alpha=0.7)
    ax.legend()

    return fig, ax
# ...
def t_sne_word_visualize(docs, labels, dimension):
    reducer = TSNE(random_state=42, n_components=dimension,
                   perplexity=40, n_iter=400)
    X = reducer.fit_transform(docs)
    return scatter_plot(X, labels, dimension)

def pca_word_visualizer(docs, labels, dimension):
    reducer = PCA(n_components=dimension)
    X = reducer.fit_transform(docs.toarray())
    return scatter_plot(X, labels, dimension)
```

All functions is in the `vizualizer.py`

### Analysis

> Process of analysis

Utilizing the methods above to make all process of research in political news. 

```python
class Analyser():

    def __init__(self, path_corpus, path_pickle, path_docs, labels_title):
        self.path_docs = path_docs
        self.path_corpus = path_corpus
        self.path_pickle = path_pickle
        self.labels_title = labels_title

        self.__init_corpus__()

    def __init_corpus__(self):
        utils.create_if_not_exists(self.path_docs)
        self.corpus = TokenizedCorpusReader(
            self.path_corpus, header=['fileid', 'subject', 'journal'])

        self.sujects = self.corpus.categ.get_cats("subject")
        self.journals = self.corpus.categ.get_cats("journal")

    def __get_docs_path(self, path):
        path = os.path.join(self.path_docs, path)
        utils.create_if_not_exists(path)
        return path

    def __get_pickle_path(self, path):
        path = os.path.join(self.path_pickle, path)
        utils.create_if_not_exists(path)
        return path
# ...
```

