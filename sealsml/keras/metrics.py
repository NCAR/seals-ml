import keras.ops as ops


def threshold_contingency_table(y_true, y_pred, threshold=0.5):
    y_pred_t = ops.convert_to_tensor(y_pred)

