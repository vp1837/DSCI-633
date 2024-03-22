import pandas as pd
import numpy as np

class my_KMeans:

    def __init__(self, n_clusters=5, n_init=10, max_iter=300, tol=1e-4):
        self.n_clusters = int(n_clusters)
        self.n_init = n_init
        self.max_iter = int(max_iter)
        self.tol = tol
        self.classes_ = range(n_clusters)
        self.cluster_centers_ = None
        self.sse_ = None

    def dist(self, a, b):
        return np.sum((np.array(a)-np.array(b))**2)**(0.5)

    def initiate(self, X):
        # Randomly initialize cluster centers
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        cluster_centers = [X[i] for i in indices]
        return cluster_centers

    def fit_once(self, X):
        cluster_centers = self.initiate(X)
        last_sse = None
        for i in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]
            sse = 0
            for x in X:
                dists = [self.dist(x, center) for center in cluster_centers]
                sse += min(dists)**2
                cluster_id = np.argmin(dists)
                clusters[cluster_id].append(x)
            if last_sse is not None and last_sse - sse < self.tol:
                break
            cluster_centers = [np.mean(cluster, axis=0) if cluster else np.nan * np.zeros_like(X[0]) for cluster in clusters]
            last_sse = sse
        return cluster_centers, sse

    def fit(self, X):
        X_feature = X.to_numpy()
        for i in range(self.n_init):
            cluster_centers, sse = self.fit_once(X_feature)
            if self.sse_ is None or sse < self.sse_:
                self.sse_ = sse
                self.cluster_centers_ = cluster_centers

    def transform(self, X):
        dists = [[self.dist(x, centroid) for centroid in self.cluster_centers_] for x in X.to_numpy()]
        return dists

    def predict(self, X):
        predictions = [np.argmin(dist) for dist in self.transform(X)]
        return predictions

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
