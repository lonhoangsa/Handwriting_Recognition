import numpy as np
from tqdm import tqdm
import pickle

def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, K=3):
        self.K = K
        self.X_train = None
        self.Y_train = None

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, X_test):
        predictions = []

        # Hiển thị tiến trình trong khi dự đoán
        for i in tqdm(range(len(X_test)), desc="Predicting"):
            dist = np.array([euc_dist(X_test[i], x_t) for x_t in self.X_train])
            k_indices = np.argsort(dist)[:self.K]
            k_nearest_labels = [self.Y_train[i] for i in k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        return predictions

    # Phương thức lưu mô hình vào tệp
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.X_train, self.Y_train), f)

    # Phương thức tải mô hình từ tệp
    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            X_train, Y_train = pickle.load(f)
        knn_model = KNN()
        knn_model.fit(X_train, Y_train)
        return knn_model
