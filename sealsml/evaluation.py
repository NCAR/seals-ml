from sklearn.metrics import accuracy_score


def compute_accuracy(y_true, y_pred):

    return accuracy_score(y_true.argmax(axis=1), y_pred[..., 0].argmax(axis=1))