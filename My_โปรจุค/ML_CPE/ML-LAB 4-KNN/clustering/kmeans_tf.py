## simple K-Means Clustering with TensorFlow (for beginners)
## Assign K-Means Clustering to TensorFlow (simple version for beginners)
## Updated: centroid update step to avoid empty cluster (if no member, keep centroid in place) 

import numpy as np

class NumpyKMeans:
    """
    K-Means Clustering implementation using pure NumPy operations.
    
    Example usage:
        km = NumpyKMeans(n_clusters=4)
        labels = km.fit_predict(X)
    """

    def __init__(self, n_clusters=4, max_iter=100, seed=42):
        self.n_clusters = n_clusters  # Number of clusters
        self.max_iter = max_iter      # Maximum number of iterations
        self.seed = seed              # Random seed for reproducibility

    # -----------------------------------------------------------------
    def _distance(self, X, centroids): #  X shape (n, d) , centroids shape (k, d)  ->  ผลลัพธ์ shape (n, k)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    # -----------------------------------------------------------------
    def fit(self, X): # Run K-Means until centroids are stable 
        X = np.asarray(X, dtype=np.float32)
        n_samples = X.shape[0]

        # Step 0: Randomly initialize centroids from existing data points
        rng = np.random.default_rng(self.seed)
        start_idx = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[start_idx].copy()

        for step in range(self.max_iter):
            # Step 1: ASSIGN - Assign points to the nearest centroid
            dist = self._distance(X, centroids)
            labels = np.argmin(dist, axis=1)

            # Step 2: UPDATE - Re-calculate centroids based on member mean
            new_centroids = np.zeros_like(centroids)
            for c in range(self.n_clusters):
                members = X[labels == c]  # Points assigned to cluster c
                if len(members) > 0:
                    new_centroids[c] = np.mean(members, axis=0)
                else:
                    # Keep existing centroid if cluster becomes empty
                    new_centroids[c] = centroids[c]

            # Step 3: CONVERGENCE CHECK - Stop if centroids no longer move significantly
            moved = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids
            if moved < 1e-4:
                break

        # Save fitted results
        dist = self._distance(X, centroids)
        self.labels_ = np.argmin(dist, axis=1)
        self.centroids_ = centroids
        self.n_iter_ = step + 1

        # Calculate inertia (Sum of squared distances of samples to their nearest cluster center)
        self.inertia_ = float(np.sum(np.min(dist, axis=1) ** 2))
        return self

    # -----------------------------------------------------------------
    def fit_predict(self, X):
        """
        Fit model and return cluster assignment labels for input data X.
        """
        return self.fit(X).labels_
