import numpy as np

def relief(X, y, k=10):
  """Computes the Relief score for each feature in the dataset.

  Args:
    X: A numpy array of shape (n_samples, n_features).
    y: A numpy array of shape (n_samples,) containing the class labels.
    k: The number of nearest neighbors to consider.

  Returns:
    A numpy array of shape (n_features,) containing the Relief scores.
  """

  n_samples, n_features = X.shape

  # Initialize the Relief scores.
  scores = np.zeros(n_features)

  # Iterate over all samples.
  for i in range(n_samples):
    # Find the nearest hit and nearest miss.
    hit = np.argmin(np.linalg.norm(X[i] - X[y == y[i]], axis=1))
    miss = np.argmin(np.linalg.norm(X[i] - X[y != y[i]], axis=1))

    # Update the Relief scores.
    for j in range(n_features):
      scores[j] += (X[i, j] - X[hit, j])**2 - (X[i, j] - X[miss, j])**2

  # Normalize the Relief scores.
  scores /= n_samples

  return scores