from sklearn.metrics import accuracy_score
import numpy as np


def compute_accuracy(y_true, y_pred):

    return accuracy_score(y_true.argmax(axis=1), y_pred[..., 0].argmax(axis=1))


def top_k_pod_accuracy(y, y_pred, k, min_thresh, max_thresh, interval):

    """ Calculate the top-k accuracy across a range of probability thresholds (POD) """

    samples = y.shape[0]
    top_k_accuracies = []
    threshold_range = np.arange(min_thresh, max_thresh + interval, interval)
    for threshold in threshold_range:

        top = np.argpartition(-y_pred, k, axis=1)[:, :k]
        top_targets = (y[np.arange(y.shape[0])[:, None], top])
        top_k_probs = y_pred[np.arange(samples)[:, None], top]
        mask = (top_k_probs > threshold).any(axis=1)
        hits = top_targets[mask].sum()
        top_k_accuracy = hits / samples
        top_k_accuracies.append(top_k_accuracy)

    return top_k_accuracies, threshold_range


def false_alarm_rate(y, y_pred, min_thresh, max_thresh, interval):

    """ Calculate the False alarm rate across probability thresholds for the top-1 probability. """

    false_alarm_rates = []
    threshold_range = np.arange(min_thresh, max_thresh + interval, interval)
    for threshold in threshold_range:

        mask = y_pred > threshold
        samples = mask.sum()
        hits = y[mask].sum()
        false_alarms = (samples - hits) / samples
        false_alarm_rates.append(false_alarms)

    return false_alarm_rates, threshold_range