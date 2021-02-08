import gensim


def generate_lsi_model(corpus, num_topics=5, type_transformation="bow"):
    dictionary = gensim.corpora.Dictionary(corpus)

    dictionary.filter_extremes(no_below=4, no_above=0.6)
    transformed_corpus = [dictionary.doc2bow(text) for text in corpus]

    if type_transformation == "tf-idf":
        tfidf = gensim.models.TfidfModel(transformed_corpus)
        transformed_corpus = tfidf[transformed_corpus]

    return gensim.models.LsiModel(
        corpus=transformed_corpus,
        id2word=dictionary,
        num_topics=num_topics,
        onepass=True,
        chunksize=500,
        power_iters=1000), dictionary, transformed_corpus


def generate_lda_model(corpus, num_topics=5, type_transformation="bow"):
    dictionary = gensim.corpora.Dictionary(corpus)

    dictionary.filter_extremes(no_below=4, no_above=0.6)
    transformed_corpus = [dictionary.doc2bow(text) for text in corpus]

    if type_transformation == "tf-idf":
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


def generate_coeherence_model(model, corpus, texts, dictionary):
    coherence_model = gensim.models.CoherenceModel(model=model,
                                                   corpus=corpus,
                                                   texts=texts,
                                                   dictionary=dictionary,
                                                   coherence='c_v')

    return coherence_model
