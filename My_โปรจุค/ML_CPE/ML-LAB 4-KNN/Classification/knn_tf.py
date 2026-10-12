import numpy as np

class NumpyKNNClassifier:

    def __init__(self, k=5):
        self.k = k  # Number of neighbors to use

    # -----------------------------------------------------------------
    def fit(self, X, y):  # Train KNN
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y, dtype=np.int32)
        self.n_classes = int(y.max()) + 1
        return self

    # -----------------------------------------------------------------
    def _distance(self, X_new):
        """
        distance = sqrt( (x1-y1)² + (x2-y2)² + ... )
        Returns distance matrix of shape (n_new, n_train)
        """
        # Calculate pair-wise Euclidean distance using broadcasting
        diff = X_new[:, None, :] - self.X_train[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    # -----------------------------------------------------------------
    def predict(self, X):
        """predict class of new data and return array of class labels"""
        X = np.array(X, dtype=np.float32)

        # Step 1: Calculate distances
        dist = self._distance(X)

        # Step 2: Select k nearest neighbors
        # np.argsort returns indices that sort distances ascendingly
        idx = np.argsort(dist, axis=1)[:, :self.k]
        neighbor_labels = self.y_train[idx]  # Shape: (n_new, k)

        # Step 3: Vote k neighbors
        # One-hot encoding using np.eye indexing
        onehot = np.eye(self.n_classes)[neighbor_labels]
        votes = np.sum(onehot, axis=1)  # Shape: (n_new, n_classes)

        return np.argmax(votes, axis=1)  # Class label with highest vote

    # -----------------------------------------------------------------
    def score(self, X, y):
        """calculate accuracy = proportion of correct predictions"""
        return float(np.mean(self.predict(X) == y))
