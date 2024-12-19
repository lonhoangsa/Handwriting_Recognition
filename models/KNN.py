
import numpy as np
from tqdm import tqdm
import pickle

def euc_dist(x1, x2, p=2):
    return np.linalg.norm(x1 - x2, ord=p)

class KNN:
    def __init__(self, k = 5, weights='uniform', algorithm='brute', p=2):
        self.K = k
        self.weights = weights
        self.algorithm = algorithm
        self.p = p
        self.X_train = None
        self.Y_train = None

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, X_test):
        predictions = []

        for i in tqdm(range(len(X_test)), desc="Predicting"):
            dist = np.array([euc_dist(X_test[i], x_t, self.p) for x_t in self.X_train])
            k_indices = np.argsort(dist)[:self.K]
            k_nearest_labels = [self.Y_train[i] for i in k_indices]

            if self.weights == 'uniform':
                most_common = np.bincount(k_nearest_labels).argmax()
            elif self.weights == 'distance':
                k_nearest_distances = dist[k_indices]
                weights = 1 / (k_nearest_distances + 1e-5)
                weighted_votes = np.bincount(k_nearest_labels, weights=weights)
                most_common = weighted_votes.argmax()

            predictions.append(most_common)
        return predictions
