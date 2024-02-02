import numpy as np
from copy import deepcopy
from collections import Counter

class my_normalizer:
    def __init__(self, norm="Min-Max", axis=1):
        self.norm = norm
        self.axis = axis

    def fit(self, X):
        self.min_vals = np.min(X, axis=self.axis)
        self.max_vals = np.max(X, axis=self.axis)

    def transform(self, X):
        X_norm = deepcopy(np.asarray(X))
        if self.norm == "Min-Max":
            for i in range(X_norm.shape[self.axis]):
                if self.axis == 0:
                    X_norm[:, i] = (X_norm[:, i] - self.min_vals[i]) / (self.max_vals[i] - self.min_vals[i])
                else:
                    X_norm[i, :] = (X_norm[i, :] - self.min_vals[i]) / (self.max_vals[i] - self.min_vals[i])
        elif self.norm == "Standard_Score":
            mean = np.mean(X, axis=self.axis)
            std_dev = np.std(X, axis=self.axis)
            for i in range(X_norm.shape[self.axis]):
                if self.axis == 0:
                    X_norm[:, i] = (X_norm[:, i] - mean[i]) / std_dev[i]
                else:
                    X_norm[i, :] = (X_norm[i, :] - mean[i]) / std_dev[i]
        elif self.norm == "L1":
            norms = np.linalg.norm(X, ord=1, axis=self.axis)
            for i in range(X_norm.shape[self.axis]):
                if self.axis == 0:
                    X_norm[:, i] = X_norm[:, i] / norms[i]
                else:
                    X_norm[i, :] = X_norm[i, :] / norms[i]
        elif self.norm == "L2":
            norms = np.linalg.norm(X, ord=2, axis=self.axis)
            for i in range(X_norm.shape[self.axis]):
                if self.axis == 0:
                    X_norm[:, i] = X_norm[:, i] / norms[i]
                else:
                    X_norm[i, :] = X_norm[i, :] / norms[i]
        return X_norm

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def stratified_sampling(y, ratio, replace=True):
    if ratio <= 0 or ratio >= 1:
        raise Exception("ratio must be 0 < ratio < 1.")
    y_array = np.asarray(y)
    unique_classes = np.unique(y_array)
    class_counts = Counter(y_array)
    sample = []
    for cls in unique_classes:
        cls_indices = np.where(y_array == cls)[0]
        num_samples_cls = int(np.ceil(ratio * class_counts[cls]))
        if replace:
            sampled_indices = np.random.choice(cls_indices, size=num_samples_cls, replace=True)
        else:
            sampled_indices = np.random.choice(cls_indices, size=num_samples_cls, replace=False)
        sample.extend(sampled_indices)
    return np.array(sample).astype(int)