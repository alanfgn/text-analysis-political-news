import atnp.categorization as categorization
import atnp.preprocessor as preprocessor
import atnp.visualizer as visualizer
import atnp.vectorizer as vectorizer
import atnp.sentiment as sentiment
import atnp.clusters as clusters
import matplotlib.pyplot as plt
import atnp.topics as topics
import atnp.utils as utils
import pandas as pd
import imgkit
import os
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from atnp.corpus import TokenizedCorpusReader
from contextlib import redirect_stdout
from collections import Counter
from joblib import dump, load
from spacy import displacy


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

    def summary(self):
        journal_count = [self.corpus.categ.get(
            fileid, "journal") for fileid in self.corpus.fileids()]
        subject_count = [self.corpus.categ.get(
            fileid, "subject") for fileid in self.corpus.fileids()]

        print(dict(Counter(journal_count)))
        print(dict(Counter(subject_count)))

    def wordcloud_subjects(self):
        base_path = self.__get_docs_path("wordcloud")

        for subject in self.sujects:
            print("Making wordcloud of %s" % subject)
            corpus = self.corpus.apply((preprocessor.word_segmentation,
                                        preprocessor.clean_tokens), categories=[subject])

            wc = visualizer.word_cloud(corpus)
            wc.to_file(os.path.join(base_path, subject + ".png"))

    def get_tokens(self, mode):
        namefile = "%s.pickle" % mode
        path = self.__get_pickle_path(os.path.join("tokens"))
        total_path = os.path.join(path, namefile)

        if not os.path.exists(total_path):
            corpus = None

            if mode == "stem":
                corpus = self.corpus.apply((preprocessor.word_segmentation,
                                            preprocessor.clean_tokens,
                                            preprocessor.remove_acentuation,
                                            preprocessor.stemmer_doc))
            elif mode == "not_stem":
                corpus = self.corpus.apply((preprocessor.word_segmentation,
                                            preprocessor.clean_tokens,
                                            preprocessor.remove_acentuation))
            elif mode == "mwe":
                corpus = self.corpus.apply((preprocessor.tokenize_mwe,
                                            preprocessor.clean_tokens,
                                            preprocessor.remove_acentuation))

            utils.save_pickle(
                path, namefile, [(list(doc), fileid) for doc, fileid in corpus])

        return utils.get_pickle(total_path)

    def gen_head_vectors(self):
        base_path = self.__get_docs_path("corpusinfo")
        corpus = self.get_tokens("stem")

        docs, fileids = zip(*corpus)

        for name, vector_fun in [("tf-idf", vectorizer.generate_tf_idf), ("frequency", vectorizer.generate_count)]:
            print("Making head for %s" % name)
            vectors, vectorizer_model = vector_fun(docs)
            df = pd.DataFrame(vectors.toarray())
            df.columns = vectorizer_model.get_feature_names()
            df.insert(0, 'Document', fileids)

            with open(os.path.join(base_path, "head-%s.txt" % name), "w") as file:
                file.write(str(df.head()))

    def reducers_with_tfidf(self):
        base_path = self.__get_docs_path("reducers")

        for mode in ["stem", "not_stem", "mwe"]:
            corpus = self.get_tokens(mode)
            self.__reducers_with_tfidf(corpus, os.path.join(base_path, mode))

    def __reducers_with_tfidf(self, corpus, base_path):
        utils.create_if_not_exists(base_path)

        docs, fileids = zip(*corpus)
        vectors, _ = vectorizer.generate_tf_idf(docs)

        for label in ['subject', 'journal']:
            y = [self.corpus.categ.get(fileid, label) for fileid in fileids]

            for dimen in [2, 3]:
                base_sub_section_path = os.path.join(base_path, "%dd" % dimen)
                utils.create_if_not_exists(base_sub_section_path)

                for name, function in [
                    ('umap', visualizer.umap_word_visualize),
                    ('t-sne', visualizer.t_sne_word_visualize),
                        ('pca', visualizer.pca_word_visualizer)]:

                    print("Saving %s %s %s" % (label, dimen, name))

                    fig, _ = function(vectors, y, dimen)
                    fig.savefig(os.path.join(base_sub_section_path,
                                             "%s-%s.png" % (label, name)))
                    plt.close(fig)

            del y

        del docs
        del fileids
        del vectors

    def word2vec_personalities(self):
        table = []
        base_path = self.__get_docs_path(
            os.path.join("word2vec"))
        pickle_path = self.__get_pickle_path(os.path.join("word2vec"))

        for theme, persons in [
            ('impeachment-dilma', ['dilma', 'lula', 'cunha', 'bolsonaro']),
            ('reforma-trabalhista', ['temer', 'maia', 'empresa', 'empregado']),
            ('afastamento-aecio', ['aecio', 'temer', 'jbs']),
            ('prisao-lula', ['lula', 'moro', ]),
            ('soltura-lula', ['lula', 'moro', 'stf', 'supremo']),
                ('reforma-previdencia', ['maia', 'guedes', "bolsonaro"])]:

            model_path_name = os.path.join(
                pickle_path, "%s-word2vec.model" % theme)
            model = None

            if not os.path.exists(model_path_name):
                print("Making model for %s" % theme)
                corpus = self.corpus.apply(
                    (preprocessor.word_segmentation,
                     preprocessor.clean_tokens,
                     preprocessor.remove_acentuation), categories=[theme])

                model = vectorizer.generate_word2vec(
                    [list(doc) for doc, fileid in corpus])
            else:
                print("Loading model %s" % theme)
                model = vectorizer.load_word2vec(model_path_name)

            df = pd.DataFrame(model.wv.vectors)
            df.insert(0, 'Token', pd.Series(model.wv.index_to_key).values)

            with open(os.path.join(base_path, "head-%s.txt" % theme), "w") as file:
                file.write(str(df.head()))

            x_vals, y_vals, labels = visualizer.reduce_dimensions_word2vec(
                model)
            fig, _ = visualizer.plot_word2vec_with_matplotlib(
                x_vals, y_vals, labels, "Word2Vec %s" % self.labels_title[theme])

            fig.savefig(os.path.join(base_path, "word2vec-%s.png" % (theme)))
            plt.close(fig)

            if not os.path.exists(model_path_name):
                model.save(model_path_name)

            for person in persons:
                print("Making table of data %s to %s" % (theme, person))
                x = [theme, person]

                for words in model.wv.most_similar(person):
                    x.extend(list(words))

                table.append(x)

            del model
            del corpus

        utils.save_csv(base_path, "personalities", table)

    def clusers_plot(self, type_model, function, tokens_type):
        base_path = self.__get_docs_path(
            os.path.join("clusters", type_model))

        corpus = self.get_tokens("stem")
        docs, fileids = zip(*corpus)
        vectors, _ = vectorizer.generate_tf_idf(docs)

        models = []
        number_clusters = []
        elbow_values = []
        silhouette_values = []
        db_values = []

        init_n_cluster = 2

        silhouette_path = os.path.join(base_path,  "silhouette")
        utils.create_if_not_exists(silhouette_path)

        for i in range(init_n_cluster, 15):
            number_clusters.append(i)
            kmeans = function(vectors, i)
            models.append(kmeans)

            silhouette_avg = silhouette_score(
                vectors.toarray(), kmeans.labels_)
            sample_silhouette_values = silhouette_samples(
                vectors.toarray(), kmeans.labels_)
            db = davies_bouldin_score(vectors.toarray(), kmeans.labels_)

            elbow_values.append(kmeans.inertia_)
            silhouette_values.append(silhouette_avg)
            db_values.append(db)

            fig_silhouette, _ = visualizer.silhouette_cluster_plot(
                vectors,
                kmeans.labels_,
                kmeans.cluster_centers_,
                i,
                silhouette_avg,
                sample_silhouette_values)

            fig_silhouette.savefig(os.path.join(
                silhouette_path, "%d_clusters.png" % i))
            plt.close(fig_silhouette)

            print("Saving silhouette of cluster %d" % i)

        choose_cluster_elbow = clusters.ellbow_optimal_number_of_clusters(
            elbow_values)

        ex_fig_silhouette_score, _ = visualizer.satter_graph_metrics(
            silhouette_values, number_clusters, "Silhouette Score")

        ex_fig_silhouette_score.show()
        choose_cluster_sihlouette = int(input("sihlouette\n"))
        plt.close(ex_fig_silhouette_score)

        ex_fig_db, _ = visualizer.satter_graph_metrics(
            db_values, number_clusters, "Davies-Bouldin Score")

        ex_fig_db.show()
        choose_cluster_db = int(input("davies bouldin\n"))
        plt.close(ex_fig_db)

        print("best elbow cluster is %d" % choose_cluster_elbow)
        print("best sihouette cluster is %d" % choose_cluster_sihlouette)
        print("best db cluster is %d" % choose_cluster_db)

        x_label = "Número de clusters"

        fig_elbow_score, _ = visualizer.satter_graph_metrics(
            elbow_values, number_clusters,
            "Método Elbow %s" % type_model,
            contrast=[number_clusters.index(choose_cluster_elbow)],
            y_label="Variação Explicada",
            x_label=x_label
        )

        fig_elbow_score.savefig(os.path.join(base_path, "elbow_score.png"))
        plt.close(fig_elbow_score)

        print("saving elbow score")

        fig_silhouette_score, _ = visualizer.satter_graph_metrics(
            silhouette_values,
            number_clusters, "Método Silhueta %s" % type_model,
            contrast=[number_clusters.index(choose_cluster_sihlouette)],
            y_label="Silhueta",
            x_label=x_label
        )

        fig_silhouette_score.savefig(os.path.join(
            base_path, "silhouette_score.png"))
        plt.close(fig_silhouette_score)

        print("saving sihouette score")

        fig_db, _ = visualizer.satter_graph_metrics(
            db_values, number_clusters,
            "Método Davies-Bouldin %s" % type_model,
            contrast=[number_clusters.index(choose_cluster_db)],
            y_label="Davies-Bouldin",
            x_label=x_label
        )

        fig_db.savefig(os.path.join(base_path, "db_score.png"))
        plt.close(fig_db)

        print("saving db score")

        chosed_clusters = list(set([choose_cluster_elbow,
                                    choose_cluster_sihlouette, choose_cluster_db]))

        labels = {"journal": [], "subject": []}

        for label in ['journal', 'subject']:
            y = [self.corpus.categ.get(fileid, label) for fileid in fileids]
            labels[label] = y

            for clst in chosed_clusters:
                model = models[number_clusters.index(clst)]
                fig, _ = visualizer.kmeans_visualizer(
                    vectors, y, model.labels_, model.cluster_centers_, "Scatter %s %d Clusters por %s" % (type_model, clst, self.labels_title[label]))

                fig.savefig(os.path.join(base_path, "%s-%d" % (label, clst)))
                print("saving labeled %s cluster %d" % (label, clst))

            del y

        pickle_path = self.__get_pickle_path(
            os.path.join("clusters", type_model))

        for clst in chosed_clusters:
            model = models[number_clusters.index(clst)]
            df = pd.DataFrame({
                "fileid": fileids,
                "cluster": model.labels_,
                "journal": labels["journal"],
                "subject": labels["subject"]})

            df.to_csv(os.path.join(
                base_path, "result-model-cluster-%d.csv" % clst))

            for label in ['journal', 'subject']:
                gr = df.groupby(["cluster", label]).size()
                gr.to_csv(
                    (os.path.join(base_path, "group-by-%s=model-cluster-%d.csv" % (label, clst))))

            utils.save_joblib(pickle_path, "df-model-cluster-%d" % clst, df)
            utils.save_joblib(
                pickle_path, "model-cluster-%s.joblib" % clst, model)

    def types_kmeans(self, tokens_type="stem"):
        self.clusers_plot("kmeans", clusters.generate_kmeans, tokens_type)
        self.clusers_plot("mini-batch-kmeans",
                          clusters.generate_mini_bath_kmeans, tokens_type)

    def mean_shift_plot(self, tokens_type="stem"):
        base_path = self.__get_docs_path(
            os.path.join("clusters",  "mean-shift"))

        corpus = self.get_tokens(tokens_type)
        docs, fileids = zip(*corpus)
        vectors, _ = vectorizer.generate_tf_idf(docs)

        mn = clusters.generate_mean_shift(vectors.toarray())

        fig, _ = visualizer.cluster_docs_plot(
            vectors, mn.labels_, mn.cluster_centers_)
        fig.savefig(os.path.join(base_path, "cluser-plot.png"))
        plt.close(fig)

        labels = {"journal": [], "subject": []}

        for label in ['journal', 'subject']:
            y = [self.corpus.categ.get(fileid, label) for fileid in fileids]
            labels[label] = y

            fig, _ = visualizer.kmeans_visualizer(
                vectors, y, mn.labels_, mn.cluster_centers_, "Scatter Mean Shift por %s" % label)

            fig.savefig(os.path.join(base_path, "%s.png" % label))
            plt.close(fig)
            del y

            print("saving labeled %s cluster" % label)

        pickle_path = self.__get_pickle_path(
            os.path.join("clusters", "mean-shift"))

        df = pd.DataFrame({
            "fileid": fileids,
            "cluster": mn.labels_,
            "journal": labels["journal"],
            "subject": labels["subject"]})

        df.to_csv(os.path.join(
            base_path, "result-model-cluster-mean-shift.csv"))

        for label in ['journal', 'subject']:
            gr = df.groupby(["cluster", label]).size()
            gr.to_csv(
                (os.path.join(base_path, "group-by-%s-model-cluster-mean-shift.csv" % (label))))

        utils.save_joblib(pickle_path, "df-model-cluster-mean-shift", df)
        utils.save_joblib(
            pickle_path, "model-cluster-mean-shift.joblib", mn)

    def afinity_plot(self, tokens_type="stem"):
        base_path = self.__get_docs_path(
            os.path.join("clusters",  "afinity"))

        corpus = self.get_tokens(tokens_type)
        docs, fileids = zip(*corpus)
        vectors, _ = vectorizer.generate_tf_idf(docs)

        af = clusters.generate_affinity_propagation(vectors.toarray())

        fig, _ = visualizer.cluster_docs_plot(
            vectors, af.labels_, af.cluster_centers_)
        fig.savefig(os.path.join(base_path, "cluser-plot.png"))
        plt.close(fig)

        labels = {"journal": [], "subject": []}

        for label in ['journal', 'subject']:
            y = [self.corpus.categ.get(fileid, label) for fileid in fileids]
            labels[label] = y

            fig, _ = visualizer.kmeans_visualizer(
                vectors, y, af.labels_, af.cluster_centers_, "Scatter Afinity por %s" % label)

            fig.savefig(os.path.join(base_path, "%s.png" % label))
            plt.close(fig)
            del y

            print("saving labeled %s cluster" % label)

        pickle_path = self.__get_pickle_path(
            os.path.join("clusters", "afinity"))

        df = pd.DataFrame({
            "fileid": fileids,
            "cluster": af.labels_,
            "journal": labels["journal"],
            "subject": labels["subject"]})

        df.to_csv(os.path.join(
            base_path, "result-model-cluster-afinity.csv"))

        for label in ['journal', 'subject']:
            gr = df.groupby(["cluster", label]).size()
            gr.to_csv(
                (os.path.join(base_path, "group-by-%s-model-cluster-afinity.csv" % (label))))

        utils.save_joblib(pickle_path, "df-model-cluster-afinity", df)
        utils.save_joblib(
            pickle_path, "model-cluster-afinity.joblib", af)

    def agglomerative_plot(self, tokens_type="stem"):
        base_path = self.__get_docs_path(
            os.path.join("clusters",  "agglomerative"))

        corpus = self.get_tokens("stem")
        docs, fileids = zip(*corpus)
        vectors, _ = vectorizer.generate_tf_idf(docs)

        pickle_path = self.__get_pickle_path(
            os.path.join("clusters", "agglomerative"))

        for algorithm in ['ward', 'complete',  'average', 'single']:
            ag = clusters.generate_agglomerative(vectors.toarray(), algorithm)
            y = [self.corpus.categ.get(fileid, "subject")
                 for fileid in fileids]

            utils.save_joblib(
                pickle_path, "model-cluster-agglomerative-%s.joblib" % algorithm, ag)

            fig, _ = visualizer.plot_dendrogram(ag, y)
            fig.savefig(os.path.join(
                base_path, "%s-dendogram.png" % algorithm))
            print("saving dendogram %s cluster" % algorithm)

    def default_categ_results(self, base_path, y_test, y_pred, name, theme, title=""):
        cfm, clr, ac = categorization.evaluating(y_test, y_pred)

        print(cfm, clr, ac)

        fig, _ = visualizer.plot_confusion_matrix(cfm, y_test, title)
        fig.savefig(os.path.join(base_path, "confusion-matix-%s-%s.png" %
                                 (name, theme)))
        plt.close(fig)

        with open(os.path.join(base_path, "report-%s-%s.txt" % (name, theme)), "w") as file:
            file.write(clr)
            file.write("\naccuracy score: %s" % str(ac))

    def analysis_nb_categ(self, tokens_type="stem"):
        base_path = self.__get_docs_path(
            os.path.join("categorization", "nb"))
        pickle_path = self.__get_pickle_path(
            os.path.join("categorization", "nb"))

        tokens = self.get_tokens(tokens_type)
        docs, fileids = zip(*tokens)
        vectors, _ = vectorizer.generate_tf_idf(docs)

        for theme in ['subject']:

            y = [self.corpus.categ.get(fileid, theme) for fileid in fileids]

            for name, alg in [("Gaussian ", categorization.generate_gaussian_nb),
                              ("Multinomial", categorization.generate_multinomial_nb),
                              ("Complement", categorization.generate_complement_nb),
                              ("Bernoulli", categorization.generate_bernoulli_nb),
                              ("Categorical", categorization.generate_categorical_nb)
                              ]:
                model, (y_test, y_pred) = alg(vectors.toarray(), y)
                utils.save_joblib(
                    pickle_path, "model-nb-%s-%s.joblib" % (theme, name), model)

                self.default_categ_results(
                    base_path, y_test, y_pred, name, theme, "Heatmap %s Naive Bayes por %s" % (name, self.labels_title[theme]))

    def analysis_svm_categ(self, tokens_type="stem"):
        base_path = self.__get_docs_path(
            os.path.join("categorization", "svm"))

        pickle_path = self.__get_pickle_path(
            os.path.join("categorization", "svm"))

        tokens = self.get_tokens(tokens_type)
        docs, fileids = zip(*tokens)
        X, _ = vectorizer.generate_tf_idf(docs)

        for theme in ['subject']:

            y = [self.corpus.categ.get(fileid, theme) for fileid in fileids]

            for kernel in ["linear", "poly", "rbf", "sigmoid"]:
                model, (y_test, y_pred) = categorization.generate_svm(
                    X.toarray(), y, kernel)

                utils.save_joblib(
                    pickle_path, "model-svm-%s-%s.joblib" % (theme, kernel), model)

                self.default_categ_results(
                    base_path, y_test, y_pred, kernel, theme, "Heatmap Svm kernel %s por %s" % (kernel, self.labels_title[theme]))

    def analysis_tree(self, tokens_type="stem"):
        base_path = self.__get_docs_path(
            os.path.join("categorization", "tree"))

        pickle_path = self.__get_pickle_path(
            os.path.join("categorization", "tree"))

        tokens = self.get_tokens(tokens_type)
        docs, fileids = zip(*tokens)
        X, _ = vectorizer.generate_tf_idf(docs)

        for theme in ['subject']:

            y = [self.corpus.categ.get(fileid, theme) for fileid in fileids]
            model, (y_test, y_pred) = categorization.generate_tree(
                X.toarray(), y)

            utils.save_joblib(
                pickle_path, "model-tree-%s.joblib" % theme, model)

            self.default_categ_results(
                base_path, y_test, y_pred, "tree", theme,  "Heatmap Tree por %s" % self.labels_title[theme])

    def analysis_nns(self, tokens_type="stem"):
        base_path = self.__get_docs_path(
            os.path.join("categorization", "nns"))

        pickle_path = self.__get_pickle_path(
            os.path.join("categorization", "nns"))

        tokens = self.get_tokens(tokens_type)
        docs, fileids = zip(*tokens)
        X, _ = vectorizer.generate_tf_idf(docs)

        for theme in ['subject']:

            for num_layers in [50, 150]:

                for num_hidden in [0, 2, 5]:
                    y = [self.corpus.categ.get(fileid, theme)
                         for fileid in fileids]

                    model, hist, (y_test, y_pred) = categorization.generate_nn(
                        X.toarray(),
                        y,
                        num_layers=num_layers,
                        num_hidden_layers=num_hidden)

                    name = "(%d)_dense-(%d)_hidden" % (num_layers, num_hidden)

                    utils.create_if_not_exists(pickle_path)
                    model.save(os.path.join(
                        pickle_path, "model-nn-%s-%s.h5" % (theme, name)))

                    self.default_categ_results(
                        base_path, y_test, y_pred, theme, name, "Heatmap Nn %s por %s" % (name, self.labels_title[theme]))

                    with open(os.path.join(base_path, "%s-%s.txt" % (theme, name)), "w") as file:
                        with redirect_stdout(file):
                            model.summary()

    def make_lsi(self, tokens, num_topics, type_transformation):
        return topics.generate_lsi_model(
            tokens, num_topics=num_topics, type_transformation=type_transformation)

    def make_lda(self, tokens, num_topics, type_transformation):
        return topics.generate_lda_model(
            tokens, num_topics=num_topics, type_transformation=type_transformation)

    def report_topics(self, model, base_path, theme, type_transformation, num_topics):
        with open(os.path.join(base_path,
                               "report-lda-%s-%d-topics-%s.txt" % (type_transformation, num_topics, theme)), "w") as f:

            for topic_id, topic in model.print_topics(num_topics=num_topics, num_words=20):
                f.write('Topic #'+str(topic_id+1)+':')
                f.write(topic + "\n")

    def analysis_topic_func(self, base_path, pickle_path, docs, func, theme, type_transformation, name, **kwargs):

        range_start = kwargs["range_start"] if "range_start" in kwargs else 2
        range_end = kwargs["range_end"] if "range_end" in kwargs else 20
        step = kwargs["step"] if "step" in kwargs else 2

        models = []
        dictionaries = []
        tcorpuses = []

        coherence_models = []
        coherence_scores = []
        topics_available = list(range(range_start, range_end, step))

        for num_topics in topics_available:

            print("Making %d tranformation %s for theme %s" %
                  (num_topics, type_transformation, theme))

            model, dictionary, transformed_corpus = func(
                docs, num_topics, type_transformation)

            coeherence_model = topics.generate_coeherence_model(
                model, transformed_corpus, docs, dictionary)

            coherence_scores.append(coeherence_model.get_coherence())

            models.append(model)
            coherence_models.append(coeherence_model)
            dictionaries.append(dictionary)
            tcorpuses.append(transformed_corpus)

        fig, _ = visualizer.coeherence_score(
            range_start, range_end, step, coherence_scores, "Coeherence for %s %s %s" % (name, type_transformation, theme))
        fig.savefig(os.path.join(
            base_path, "coeherence-%s-%s.png" % (type_transformation, theme)))

        fig.show()
        choose_num_topic = int(input("Num Topic\n"))
        plt.close(fig)

        calculated_index_topic = int((choose_num_topic - range_start)/step)

        choosed_model = models[calculated_index_topic]
        coherence_model = coherence_models[calculated_index_topic]
        dictionary = dictionaries[calculated_index_topic]
        tcorpus = tcorpuses[calculated_index_topic]
        coherence_score = coherence_scores[calculated_index_topic]

        with open(os.path.join(base_path, "coeherence-topics-%s-%s-%s.txt" % (name, theme, type_transformation)), "w") as f:
            f.write(str(coherence_score))

        utils.create_if_not_exists(pickle_path)
        choosed_model.save(os.path.join(
            pickle_path, "model-%s-%s-%s.model" % (name, theme, type_transformation)))
        coherence_model.save(os.path.join(
            pickle_path, "coherence-model-%s-%s-%s.model" % (name, theme, type_transformation)))
        dictionary.save(os.path.join(
            pickle_path, "dictionary-%s-%s-%s.dict" % (name, theme, type_transformation)))
        tcorpus.save(os.path.join(pickle_path, "tcorpus-%s-%s-%s.model" %
                                  (name, theme, type_transformation)))

        calculated_num_topics = topics_available[calculated_index_topic]
        print("Calculated num topic available is %d" %
              calculated_num_topics)

        self.report_topics(choosed_model, base_path, theme,
                           type_transformation, calculated_num_topics)

        del models
        del coherence_scores
        del docs

    def analysis_topic(self, func, name, transformations=["tf-idf", "bow"]):
        base_path = self.__get_docs_path(os.path.join("topics", name))
        pickle_path = self.__get_pickle_path(
            os.path.join("topics", name))

        tokens = self.get_tokens("mwe")
        docs, _ = zip(*tokens)

        range_start = 4
        range_end = 6
        step = 2

        for type_transformation in transformations:
            self.analysis_topic_func(base_path, pickle_path, docs, func, "all", type_transformation, name,
                                     range_start=range_start, range_end=range_end, step=step)

            for theme in self.sujects:
                print("making model for subject %s" % theme)

                corpus = self.corpus.apply((preprocessor.tokenize_mwe,
                                            preprocessor.clean_tokens,
                                            preprocessor.remove_acentuation), categories=[theme])

                self.analysis_topic_func(base_path, pickle_path, [list(doc) for doc, fileid in corpus], func, theme, type_transformation, name,
                                         range_start=range_start, range_end=range_end, step=step)

                del corpus

    def analysis_lda(self, transformations=["tf-idf"]):
        self.analysis_topic(self.make_lda, "lda", transformations)

    def analysis_lsi(self, transformations=["tf-idf"]):
        self.analysis_topic(self.make_lsi, "lsi", transformations)

    def semantic(self):
        base_path = self.__get_docs_path("semantic")
        nlp = preprocessor.SpacyInstance.get_normal_instance()

        sentences = ""
        pos_sentences = ""

        for theme in self.sujects:
            print("Making Semantic for %s" % theme)
            document = next(self.corpus.documents(categories=[theme]))
            ndoc = nlp(document[0])

            ehtml = displacy.render(ndoc, style='ent')

            meta = "<meta http-equiv='Content-Type' content='text/html; charset=UTF-8'>"
            ehtml = "%s%s" % (meta, ehtml)

            imgkit.from_string(ehtml, os.path.join(
                base_path, "%s-random_entity-%s.jpg" % (theme, document[1])), options={"encoding": "UTF-8"})

            sent = list(ndoc.sents)[0]
            dephtml = displacy.render(
                sent, style="dep")

            dephtml = "%s%s" % (meta, dephtml)

            sentences += "%s%s" % (str(sent), "\n")
            pos_sentences += "%s%s" % (str([(w, w.pos_) for w in sent]), "\n")

            imgkit.from_string(dephtml, os.path.join(
                base_path, "%s-dept-%s.jpg" % (theme, document[1])), options={"encoding": "UTF-8"})

        with open(os.path.join(base_path, "sentences.txt"), "w") as file:
            file.write(sentences)

        with open(os.path.join(base_path, "pos_sentences.txt"), "w") as file:
            file.write(pos_sentences)

    def analysis_sentiment(self):
        base_path = self.__get_docs_path("sentiment")
        pickle_path = self.__get_pickle_path(
            os.path.join("sentiment"))

        tokens = self.get_tokens("mwe")

        sent_corpus = []
        count = 0
        df = None
        df_path = os.path.join(pickle_path, "df-sentiment")

        if not os.path.exists(df_path):
            for doc, fileid in tokens:
                count = count + 1

                print("fileid:", count)
                print((count/1706) * 100, "%")

                opl_cnt = Counter(sentiment.categs_pol_oplexicon_document(doc))
                sent_cnt = Counter(sentiment.categs_pol_sentilex_document(doc))

                sent_corpus.append((
                    fileid,
                    self.corpus.categ.get(fileid, 'subject'),
                    self.corpus.categ.get(fileid, 'journal'),
                    sent_cnt[None],
                    sent_cnt[-1],
                    sent_cnt[1],
                    sent_cnt[0],
                    opl_cnt[None],
                    opl_cnt[-1],
                    opl_cnt[1],
                    opl_cnt[0]
                ))

            df = pd.DataFrame(sent_corpus, columns=[
                'fileid',
                'subject',
                'journal',
                'sent_none',
                'sent_neg',
                'sent_pos',
                'sent_neut',
                'opl_none',
                'opl_neg',
                'opl_pos',
                'opl_neut'
            ])

            df['polarity_oplexicon'] = df['opl_pos'] - df['opl_neg']
            df['polarity_sentilex'] = df['sent_pos'] - df['sent_neg']

            df.to_pickle(os.path.join(pickle_path, "df-sentiment"))
        else:
            print("Loading sentiment df")
            df = pd.read_pickle(df_path)

        # df.to_csv(os.path.join(base_path, "df-sentiments.csv"))

        dfopl = df[['subject', 'journal', 'polarity_oplexicon',
                    'opl_neg', 'opl_pos', 'opl_neut', 'opl_none']]
        dfsent = df[['subject', 'journal', 'polarity_sentilex',
                     'sent_neg', 'sent_pos', 'sent_neut', 'sent_none']]

        for theme in ['subject', 'journal']:
            gdfopl = dfopl.groupby(theme).sum()
            gdfsent = dfsent.groupby(theme).sum()

            gdfopl.to_csv(os.path.join(
                base_path, 'opl-groupby-%s.csv' % theme))
            gdfsent.to_csv(os.path.join(
                base_path, 'sent-groupby-%s.csv' % theme))

            fig, _ = visualizer.plot_compare_bar(
                ('Oplexicon', list(gdfopl['polarity_oplexicon'].values)),
                ('Sentilex', list(gdfsent['polarity_sentilex'].values)),
                gdfopl['polarity_oplexicon'].to_dict().keys(), "Comparativo de polaridade por lexicos e por %s" % self.labels_title[theme])

            fig.savefig(os.path.join(
                base_path, "compare_bar_plot_%s.png" % theme))

            fig, _ = visualizer.plot_compare_bar(
                ('Oplexicon', list(gdfopl['opl_none'].values)),
                ('Sentilex', list(gdfsent['sent_none'].values)),
                gdfsent['sent_none'].to_dict().keys(), "Comparativo de perda por lexicos e por %s" % self.labels_title[theme], "Quantidade")

            fig.savefig(os.path.join(
                base_path, "missing_compare_bar_plot_%s.png" % theme))

            for name, _gdf in [('Oplexicon', gdfopl), ('Sentliex', gdfsent)]:
                gdf = _gdf.copy()

                gdf.columns = ['pol', 'neg', 'pos', 'neut', 'none']
                tgdf = gdf.transpose()

                result = {key: [value['neg'], value['neut'], value['pos']]
                          for key, value in tgdf.to_dict().items()}

                fig, _ = visualizer.compare_survey(
                    result, ['Negativo', 'Neutro', 'Positivo'], "Distribuição Polaridade %s por %s" % (name, theme))
                fig.savefig(os.path.join(
                    base_path, "comapre_dist_bar_plot_%s_%s.png" % (theme, name)))
