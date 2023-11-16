import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from hagelslag.evaluation.ContingencyTable import ContingencyTable
from hagelslag.evaluation.ProbabilityMetrics import DistributedROC, DistributedReliability, DistributedCRPS

def provide_metrics(y_true, probabilities):

    metrics = {}

    y_pred = np.zeros(shape=probabilities.shape)
    y_pred[np.arange(y_pred.shape[0]), np.argmax(probabilities, axis=1)] = 1

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["roc_auc_ovr"] = roc_auc_score(y_true, probabilities)

    return metrics

