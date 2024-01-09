import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def stats(predictions, actual, label='', print_out=False):
  total = len(actual)
  correct = 0
  classes = np.unique(actual)
  conf_matrix = {a: {b: 0 for b in classes} for a in classes}

  for p, a in zip(predictions, actual):
    correct += p == a
    conf_matrix[p][a] += 1

  accuracy = correct / total
  if print_out:
    if label: print(label.upper() + ':')
    print('Accuracy: ' + str(accuracy))
    print(conf_matrix)
  return [accuracy, conf_matrix]


def best_bound(start, end, interval, model, X_train, y_train, X_test, y_test, display_graph=False, args={}):
  possible = []
  i = 0
  while start + i * interval < end:
    possible.append(start + i * interval)
    i += 1
  accuracy = [0] * len(possible)
  for i, x in enumerate(possible):
    accuracy[i] = stats(model(X_train, y_train, X_test, x, **args), y_test)[0]
  if display_graph: plt.plot(np.array(possible), np.array(accuracy))
  return possible[np.argmax(accuracy)]


def visualize2Ddata(X, y, colors):
  df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
  fig, ax = plt.subplots()
  grouped = df.groupby('label')
  for key, group in grouped:
      group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
  plt.show()


def best_model(models, X_train, y_train, X_test, y_test, print_out=False):
  best_accuracy = 0
  best_models = []

  for model_name in models:
    model = models[model_name][0]
    args = {} if len(models[model_name]) == 1 else {'dist_metric': models[model_name][1]}

    x = best_bound(1, 75, 1, model, X_train, y_train, X_test, y_test, args)
    accuracy = stats(model(X_train, y_train, X_test, x, **args), y_test)[0]

    if abs(accuracy - best_accuracy) < 1e-9:
      best_models.append(model_name)
    elif accuracy > best_accuracy:
      best_models = [model_name]
      best_accuracy = accuracy

    if print_out: print(model_name + ': ' + str(accuracy))

  return [best_models, best_accuracy]