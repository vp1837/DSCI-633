import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan"}
        # p value only matters when metric = "minkowski"
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y
        return

    def dist(self, x):
        # Calculate distances of training data to a single input data point (distances from self.X to x)
        # Output np.array([distances to x])
        if self.metric == "minkowski":
            distances = np.power(np.sum(np.power(np.abs(self.X - x), self.p), axis=1), 1/self.p)

        elif self.metric == "euclidean":
            distances = np.sqrt(np.sum(np.power((self.X - x), 2), axis=1))

        elif self.metric == "manhattan":
            distances = np.sum(np.abs(self.X - x), axis=1)

        else:
            raise Exception("Unknown criterion.")
        return distances

    def k_neighbors(self, x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors) e.g. {"Class A":3, "Class B":2}
        distances = self.dist(x)
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:self.n_neighbors]
        k_nearest_labels = [self.y[i] for i in k_nearest_indices]
        output = Counter(k_nearest_labels)
        return output

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        predictions = []
        for _, x in X.iterrows():
            neighbors = self.k_neighbors(x)
            predicted_label = max(neighbors, key=neighbors.get)
            predictions.append(predicted_label)
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Input data mismatch.")

        for x in X_feature.to_numpy():
            neighbors = self.k_neighbors(x)
            # Calculate the probability of data point x belonging to each class
            # e.g. prob = {"2": 1/3, "1": 2/3}
            prob = {cls: neighbors[cls] / self.n_neighbors for cls in self.classes_}
            probs.append(prob)
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs
