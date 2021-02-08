import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
import math
import os
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from matplotlib import rcParams
from wordcloud import WordCloud
import random

sns.set()

cmap = cm.get_cmap("viridis")
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Ubuntu']


def word_cloud(corpus):
    text = ""
    for document, _ in corpus:
        text += " ".join(document)

    # Circle mask
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)

    return WordCloud(scale=10,
                     mask=mask,
                     colormap=cmap,
                     background_color='white',
                     min_font_size=8).generate(text)


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


def scatter_plot_3d(x, labels):
    fig, ax = plt.subplots(subplot_kw={"projection": '3d'})
    fig.set_size_inches(12, 7)

    df = pd.DataFrame(x, columns=['x', 'y', 'z'])
    df['label'] = labels

    for idx, label in enumerate(set(labels)):
        ax.scatter(
            xs=df[df['label'] == label]["x"],
            ys=df[df['label'] == label]["y"],
            zs=df[df['label'] == label]["z"],
            label=label,
        )

    ax.legend()
    return fig, ax


def scatter_plot(x, labels, dimension):
    if dimension == 2:
        return scatter_plot_2d(x, labels)

    if dimension == 3:
        return scatter_plot_3d(x, labels)


def umap_word_visualize(docs, labels, dimension):
    reducer = umap.UMAP(random_state=42, n_components=dimension)
    X = reducer.fit_transform(docs)

    return scatter_plot(X, labels, dimension)


def t_sne_word_visualize(docs, labels, dimension):
    reducer = TSNE(random_state=42, n_components=dimension,
                   perplexity=40, n_iter=400)
    X = reducer.fit_transform(docs)
    return scatter_plot(X, labels, dimension)


def pca_word_visualizer(docs, labels, dimension):
    reducer = PCA(n_components=dimension)
    X = reducer.fit_transform(docs.toarray())

    return scatter_plot(X, labels, dimension)


def kmeans_visualizer(docs, docs_labels, docs_clusters, centers, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 7)

    reducer = PCA(n_components=2)
    X = reducer.fit_transform(docs.toarray())
    centroids = reducer.transform(centers)

    df = pd.DataFrame(X, columns=['x', 'y'])
    df['label'] = docs_labels

    for idx, cluster in enumerate(docs_clusters):
        ax.plot(
            [centroids[cluster, 0], X[idx, 0]], [
                centroids[cluster, 1], X[idx, 1]],
            linestyle='-', color="#444444", linewidth=1, alpha=0.2, zorder=2)

    for idx, label in enumerate(set(docs_labels)):
        values = df[df['label'] == label]
        ax.scatter(values["x"], values["y"], marker=".", label=label, zorder=1)

    ax.scatter(centroids[:, 0], centroids[:, 1],
               marker='o', c="w", alpha=1, s=200, zorder=3)

    for idx in range(len(centroids)):
        ax.scatter(centroids[idx, 0], centroids[idx, 1], color="k",
                   marker="$%d$" % idx, zorder=4)

    ax.legend()
    ax.set_title(title)

    return fig, ax


def silhouette_plot(y, n_clusters, silhouette_avg, sample_silhouette_values, ax):
    y_lower = 10

    ax.set_ylim([0, len(y) + (n_clusters + 1) * 10])
    ax.set_xlim([-0.1, 1])

    for cluster in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[y == cluster]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cmap(float(cluster) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7)

        ax.text(-0.15, y_lower + 0.5 * size_cluster_i, str(cluster))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="-", alpha=0.8)

    ax.set_yticks([])

    return ax


def cluster_plot(x, y, centers, ax=None):
    colors = cmap(y.astype(float) / len(centers))

    ax.scatter(x[:, 0], x[:, 1], marker='.', s=30,
               lw=0, alpha=0.7, c=colors, edgecolor='k')

    ax.scatter(centers[:, 0], centers[:, 1],
               marker='D', c="w", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax.scatter(c[0], c[1], marker='$%d$' %
                   i, alpha=1, s=50, edgecolor='k')

    return ax


def cluster_docs_plot(docs, y, centers):
    reducer = PCA(n_components=2)
    X = reducer.fit_transform(docs.toarray())

    centroids = reducer.transform(centers)

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 7)

    ax = cluster_plot(X, y, centroids, ax)

    return fig, ax


def silhouette_cluster_plot(docs, y, centers, n_clusters, silhouette_avg, sample_silhouette_values):
    reducer = PCA(n_components=2)
    X = reducer.fit_transform(docs.toarray())
    centroids = reducer.transform(centers)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 10)

    silhouette_plot(y, n_clusters, silhouette_avg,
                    sample_silhouette_values, ax1)
    cluster_plot(X, y, centroids, ax2)

    return fig, (ax1, ax2)


def satter_graph_metrics(points, clusters, title, color=1, contrast=[], y_label="", x_label=""):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)

    ax.plot(clusters, points, linestyle="-",
            color=cmap(color), marker="o", zorder=1)

    if len(contrast) > 0:
        ax.scatter([clusters[i] for i in contrast], [points[i]
                                                     for i in contrast], marker="o", color="r", s=120, zorder=2)

        ax.axvline(x=[clusters[i] for i in contrast], color="r",
                   linestyle="-", alpha=0.8, zorder=3)

    ax.set_title(title)
    ax.set_xticks(clusters)

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    return fig, ax


def plot_dendrogram(model, y):

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, sharex=False, sharey=False,
        gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.002})
    fig.set_size_inches(25, 10)

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]).astype(float)

    colormap = {label: cmap(idx / len(set(y)))
                for idx, label in enumerate(set(y))}

    ddata = dendrogram(linkage_matrix, labels=y, no_labels=True, ax=ax1)
    colors = [colormap[label] for label in ddata['ivl']]

    ax2.bar([idx for idx, _ in enumerate(y)],  [
            1 for _ in range(len(y))], color=colors, edgecolor=colors)

    ax2.set_xlim(-0.5, len(y)-.5)

    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.legend(handles=[mpatches.Patch(color=b, label=a) for a, b in colormap.items()],
               bbox_to_anchor=(0.5, -0.5), loc='lower center', ncol=len(colormap.values()))

    return fig, (ax1, ax2)


def plot_confusion_matrix(cm, y_true, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)

    df_cm = pd.DataFrame(cm, columns=np.unique(
        y_true), index=np.unique(y_true))
    df_cm.index.name = 'Real'
    df_cm.columns.name = 'Previsto'

    sns.heatmap(df_cm, cmap=cmap, annot=True, ax=ax)
    ax.tick_params(labelrotation=0)
    ax.set_title(title)

    return fig, ax


def plot_compare_bar(y1, y2, labels, title="", ylabel="Polaridade"):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)

    x = np.arange(len(labels))
    width = 0.35

    category_colors = cmap(np.linspace(0.15, 0.85, 2 ))

    rects1 = ax.bar(x - width/2, y1[1], width, label=y1[0], color=category_colors[0])
    rects2 = ax.bar(x + width/2, y2[1], width, label=y2[0], color=category_colors[1])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    return fig, ax


def compare_survey(results, category_names, title):

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = cmap(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)

    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color, linewidth=0)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    
    ax.set_title(title)

    return fig, ax


def coeherence_score(range_start, range_end, step, coherence_scores, title):
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)

    x_ax = range(range_start, range_end, step)
    y_ax = coherence_scores

    ax.plot(x_ax, y_ax, c='r')
    ax.axhline(y=sum(coherence_scores)/len(coherence_scores),
               c='k', linestyle='--', linewidth=2)

    ax.set_xlabel('Numero de topicos')
    ax.set_ylabel('Coerência')
    ax.set_title(title)

    return fig, ax


def reduce_dimensions_word2vec(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_word2vec_with_matplotlib(x_vals, y_vals, labels, title):
    random.seed(0)
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)

    ax.scatter(x_vals, y_vals)

    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)

    for i in selected_indices:
        ax.annotate(labels[i], (x_vals[i], y_vals[i]))

    ax.set_title(title)
    return fig, ax
