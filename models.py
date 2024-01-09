import numpy as np
from collections import Counter
from scipy.spatial import distance


def knn(X_train, y_train, X_test, k, dist_metric='euclidean'):
  # Compute distances between each test point and each training point then sort by distances
  distances = distance.cdist(X_test, X_train, dist_metric)
  sorted_indices = np.argsort(distances, axis=1)

  # Return the predictions for each test point based on the labels of its nearest neighbors.
  predictions = np.array([Counter([y_train[i] for i in point[:k]]).most_common(1)[0][0] for point in sorted_indices])
  return predictions


def tnn(X_train, y_train, X_test, t, dist_metric='euclidean'):
  # Compute distances between each test point and each training point then sort by distances
  distances = distance.cdist(X_test, X_train, dist_metric)
  sorted_indices = np.argsort(distances, axis=1)

  # Return the predictions for each test point based on the labels of its nearest neighbors.
  predictions = []
  for i, point in enumerate(sorted_indices):
    d = 0
    nearest = []
    j = 0
    while d < t:
      d += distances[i][point[j]]
      nearest.append(y_train[point[j]])
      j += 1
    predictions.append(Counter(nearest).most_common(1)[0][0])
  return np.array(predictions)


def fixed_radius(X_train, y_train, X_test, r, dist_metric='euclidean'):
  # Compute distances between each test point and each training point then sort by distances
  distances = distance.cdist(X_test, X_train, dist_metric)
  sorted_indices = np.argsort(distances, axis=1)

  # Return the predictions for each test point based on the labels of its nearest neighbors.
  predictions = []
  for i, point in enumerate(sorted_indices):
    nearest = []
    for j in range(len(point)):
      nearest.append(y_train[point[j]])
      if distances[i][point[j]] > r: break
      # nearest.append(y_train[point[j]])
    predictions.append(Counter(nearest).most_common(1)[0][0])
  return np.array(predictions)