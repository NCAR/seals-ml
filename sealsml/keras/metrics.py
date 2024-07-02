import keras.ops as ops
import keras


@keras.saving.register_keras_serializable()
def mean_searched_locations(y_true, y_pred):
    """
    Calculates the mean number of leak locations that need to be searched before finding the true leak
    assuming the leak locations are sorted by the predicted probability of a leak at that location.
    Minimum mean_search_locations should be 1. 

    Args:
        y_true: Tensor or array of shape [batch_size, num_locations] or [batch_size, num_locations, num_outputs]
        y_pred: Tensor or array of shape [batch_size, num_locations] or [batch_size, num_locations, num_outputs]

    Returns:
        expected_searched_locations
    """
    pred_search_length = search_length(y_true, y_pred)
    return ops.mean(pred_search_length) + 1.0 # Add 1 to account for 0-based indexing.

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
    leak_index = ops.argmax(y_true, axis=1)
    pred_leak_loc_order = ops.argsort(y_pred, axis=1)[:, ::-1]
    pred_search_length = ops.argmin(ops.abs(pred_leak_loc_order - ops.expand_dims(leak_index, axis=-1)), axis=1)
    return pred_search_length


