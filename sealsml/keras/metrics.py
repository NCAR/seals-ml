import keras.ops as ops
import keras


@keras.saving.register_keras_serializable()
def mean_searched_locations(y_true, y_pred):
    """
    Calculates the mean number of leak locations that need to be searched before finding the true leak
    assuming the leak locations are sorted by the predicted probability of a leak at that location.

    Args:
        y_true: Tensor or array of shape [batch_size, num_locations] or [batch_size, num_locations, num_outputs]
        y_pred: Tensor or array of shape [batch_size, num_locations] or [batch_size, num_locations, num_outputs]

    Returns:
        expected_searched_locations
    """
    pred_search_length = search_length(y_true, y_pred)
    return ops.mean(pred_search_length)

@keras.saving.register_keras_serializable()
def search_length(y_true, y_pred):
    """
    Calculates how many locations have to be searched before finding the true leak for each example.

    Args:
        y_true:
        y_pred:

    Returns:
        keras Tensor containing the search length count for each example.
    """
    y_true_2d = ops.squeeze(y_true)
    y_pred_2d = ops.squeeze(y_pred)
    leak_index = ops.argmax(y_true_2d, axis=1)
    pred_leak_loc_order = ops.argsort(y_pred_2d, axis=1)[:, ::-1]
    pred_search_length = []
    for i in range(ops.shape(leak_index)[0]):
        pred_search_length.append(ops.where(pred_leak_loc_order[i] == leak_index[i])[0][0])
    return ops.stack(pred_search_length)
