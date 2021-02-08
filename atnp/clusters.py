import collections
import math
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering

def generate_kmeans(corpus, clusters):
    
    model = KMeans(n_clusters=clusters, max_iter=500)
    model.fit_transform(corpus)

    return model

def generate_mini_bath_kmeans(corpus, clusters):
    
    model = MiniBatchKMeans(
        n_clusters=clusters, 
        random_state=0, 
        batch_size=6, 
        max_iter=500)
    
    model.fit(corpus)

    return model

def generate_mean_shift(corpus):
    
    bandwidth = estimate_bandwidth(corpus, quantile=0.3)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(corpus)

    return ms

def generate_affinity_propagation(corpus):
    af = AffinityPropagation(max_iter=800, random_state=0, convergence_iter=30, verbose=True)
    af.fit(corpus)

    return af

def generate_agglomerative(corpus, linkage):
    ag = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=linkage)
    ag.fit(corpus)

    return ag

def ellbow_optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2