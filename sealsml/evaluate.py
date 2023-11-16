import numpy as np
from sklearn.metrics import accuracy_score

def provide_metrics(y_true, probabilities):

    metrics = {}

    y_pred = np.zeros(shape=probabilities.shape)
    y_pred[np.arange(y_pred.shape[0]), np.argmax(probabilities, axis=1)] = 1

    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    return metrics

