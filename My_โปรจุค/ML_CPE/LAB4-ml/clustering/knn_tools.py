## KNN with mean function (for clustering) 

import numpy as np

class KNNClusterAssigner:

    def __init__(self, k=5):
        self.k = k

    # -----------------------------------------------------------------
    def fit(self, X, cluster_labels):

        self.X = np.array(X, dtype=np.float32)
        self.labels = np.array(cluster_labels, dtype=np.int32)
        self.n_clusters = int(cluster_labels.max()) + 1
        return self

    # -----------------------------------------------------------------
    def predict(self, X_new):
  
        X_new = np.array(X_new, dtype=np.float32)

        # 1) Calculate Euclidean distance from new points to all training points
        diff = X_new[:, None, :] - self.X[None, :, :]
        dist = np.sqrt(np.sum(np.square(diff), axis=2))

        # 2) Select the k nearest neighbors (smallest distances)
        _, idx = np.argpartition(-dist, kth=self.k, axis=1)
        neighbor_labels = self.labels[idx]

        # 3) Majority voting: Count cluster frequency among k neighbors
        onehot = np.eye(self.n_clusters)[neighbor_labels]
        votes = np.sum(onehot, axis=1)

        return np.argmax(votes, axis=1).astype("int32")
