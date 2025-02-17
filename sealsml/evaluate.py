import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sealsml.data import Preprocessor
import xarray as xr
from sealsml.geometry import get_relative_azimuth
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def provide_metrics(y_true, probabilities):

    metrics = {}

    y_pred = np.zeros(shape=probabilities.shape)
    y_pred[np.arange(y_pred.shape[0]), np.argmax(probabilities, axis=1)] = 1

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["roc_auc_ovr"] = roc_auc_score(y_true, probabilities)

    return metrics


def plot_probability_map(model, scaler, les_path, validation_file, sample_number):
    validation_data = xr.open_dataset(validation_file)
    file_name = validation_file.replace("training_data_", "").replace(".nc", "").split('/')[-1]
    raw_data = xr.open_dataset(os.path.join(les_path, file_name))
    LES_data = raw_data.assign_coords(
        {'iDim': raw_data['xPos'].values[0, 0, :],
         'jDim': raw_data['yPos'].values[0, :, 0],
         'kDim': raw_data['zPos'].values[:, 0, 0]})
    arr = np.zeros(shape=(900, 8))
    met_locs = validation_data['met_sensor_loc'].isel(sample=sample_number).values
    u = LES_data['u'].isel(timeDim=slice(0, 100), kDim=1).mean(dim='timeDim').values.flatten().reshape(1, -1)
    v = LES_data['v'].isel(timeDim=slice(0, 100), kDim=1).mean(dim='timeDim').values.flatten().reshape(1, -1)
    x = LES_data['xPos'].values[0].flatten()
    y = LES_data['yPos'].values[0].flatten()
    z = LES_data['zPos'].values[0].flatten()
    ch4 = LES_data['q_CH4'].isel(time=slice(0, 100), kDim=1).mean(dim='time').values.flatten()
    derived_vars = get_relative_azimuth(u=u,
                                        v=v,
                                        x_ref=met_locs[0],
                                        y_ref=met_locs[1],
                                        z_ref=met_locs[2],
                                        x_target=x,
                                        y_target=y,
                                        z_target=z,
                                        time_series=False).T

    arr[:, :6] = derived_vars
    decoder = arr.reshape(1, 900, 1, 8, 1)
    p = Preprocessor(scaler_type="quantile", sensor_pad_value=-1, sensor_type_value=-999)
    p.scaler = scaler
    encoder_data, decoder_data, targets = p.load_data([validation_file])
    decoder = xr.DataArray(decoder)
    scaled_encoder, encoder_mask = p.preprocess(encoder_data, fit_scaler=False)
    scaled_decoder, decoder_mask = p.preprocess(decoder, fit_scaler=False)
    probabilities = model.predict(x=(
        scaled_encoder[sample_number:sample_number + 1], np.expand_dims(scaled_decoder[..., :4], axis=0),
        encoder_mask[sample_number:sample_number + 1]),
        batch_size=1024).squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')

    leak_mask = validation_data['decoder_input'][sample_number, :, 0, 0, -1].values > -1
    sensor_mask = validation_data['encoder_input'][sample_number, :, 0, 0, -1].values > -1
    pot_leaks = validation_data['leak_meta'].values[sample_number, :, :2][leak_mask]
    sensors = validation_data['sensor_meta'].values[sample_number, :, :2][sensor_mask]
    met_loc = validation_data['met_sensor_loc'].values[sample_number, :2]
    # sensor_ch4 = validation_data['encoder_input'][sample_number, :, 0, -1, 0][sensor_mask].values
    target = validation_data['target'][sample_number].values.argmax()

    ch = axes[0].pcolormesh(LES_data['xPos'][0, 0, :].values, LES_data['yPos'][0, :, 0].values, ch4.reshape(30, 30),
                            norm=LogNorm(vmin=1e-8, vmax=1e-4), cmap='Purples')
    axes[0].scatter(pot_leaks[:, 0], pot_leaks[:, 1], s=100, color='blue', edgecolor='k')
    axes[0].scatter(sensors[:, 0], sensors[:, 1], s=100, color='red', edgecolor='k')
    axes[0].scatter(met_loc[0], met_loc[1], color='k', marker="2", s=1000)
    axes[0].set_title('Mean CH4 Plume')
    plt.colorbar(ch, ax=axes[0])

    probs = axes[1].pcolormesh(LES_data['xPos'][0, 0, :].values, LES_data['yPos'][0, :, 0].values,
                               probabilities.reshape(30, 30), vmin=0, vmax=1, cmap='Greys')
    # axes[1].scatter(pot_leaks[0, 0], pot_leaks[0, 1], s=200, color='white', edgecolor='k', marker='s', label='Max Probability')
    axes[1].scatter(pot_leaks[:, 0], pot_leaks[:, 1], s=100, color='blue', edgecolor='k',
                    label='Potential Leak Location')
    axes[1].scatter(sensors[:, 0], sensors[:, 1], s=100, color='red', edgecolor='k', label='CH4 Sensor')
    axes[1].scatter(pot_leaks[target, 0], pot_leaks[target, 1], s=100, color='white', edgecolor='k', marker='*',
                    label='Leak Location')
    axes[1].scatter(met_loc[0], met_loc[1], color='k', marker="2", s=500, label='Met Sensor')
    fig.legend(loc='outside right upper')
    axes[1].set_title('Probabilities')
    plt.colorbar(probs, ax=axes[1])


def calculate_distance_matrix(array: np.ndarray, export_matrix: bool = False) -> tuple:
    """
    Calculate the distance matrix for an array of 3D points.

    Parameters:
        array (np.ndarray): An array of shape (N, 3) containing x, y, z coordinates of points.
        export_matrix (bool, optional): If True, the distance matrix will be returned along with
            the minimum, median, and maximum distances. If False (default), only the minimum,
            median, and maximum distances will be returned.

    Returns:
        tuple or np.ndarray: If export_matrix is False, a tuple containing the minimum, median,
            and maximum distances. If export_matrix is True, a tuple containing the distance
            matrix along with the minimum, median, and maximum distances.

    Raises:
        ValueError: If the input array is not 2D, or the resulting distance matrix is not square,
        or if the number of points in the input array does not match the size of the distance matrix.
    """
    # Check if the input array is 2D
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")

    # Reshape array to have new axes for broadcasting
    array_reshaped1 = array[:, np.newaxis, :]
    array_reshaped2 = array[np.newaxis, :, :]

    # Calculate the differences between corresponding points
    differences = array_reshaped1 - array_reshaped2

    # Calculate the distances along the last axis (axis=2)
    distance_matrix = np.linalg.norm(differences, axis=2)

    # Check if the distance matrix is square
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Output distance array must be square")

    # Check if the number of points in the input array matches the size of the distance matrix
    if array.shape[0] != distance_matrix.shape[0]:
        raise ValueError("Number of points in input array must match size of distance matrix")

    # Calculate min, median, and max distances excluding zeros
    non_zero_values = distance_matrix[distance_matrix != 0]
    matrix_min = np.min(non_zero_values)
    matrix_median = np.median(non_zero_values)
    matrix_max = np.max(non_zero_values)

    if export_matrix:
        return distance_matrix, matrix_min, matrix_median, matrix_max
    else:
        return matrix_min, matrix_median, matrix_max

def compute_accuracy(y_true, y_pred):

    return accuracy_score(y_true.argmax(axis=1), y_pred[..., 0].argmax(axis=1))


def top_k_pod_accuracy(y, y_pred, k, min_thresh, max_thresh, interval):

    """ Calculate the top-k accuracy across a range of probability thresholds (POD) """

    samples = y.shape[0]
    threshold_range = np.arange(min_thresh, max_thresh + interval, interval)
    top_k_accuracies = np.zeros(shape=threshold_range.shape)
    for i, threshold in enumerate(threshold_range):

        top = np.argpartition(-y_pred, k, axis=1)[:, :k]
        top_targets = (y[np.arange(y.shape[0])[:, None], top])
        top_k_probs = y_pred[np.arange(samples)[:, None], top]
        mask = (top_k_probs > threshold).any(axis=1)
        hits = top_targets[mask].sum()
        top_k_accuracy = hits / samples
        top_k_accuracies[i] = top_k_accuracy

    return top_k_accuracies, threshold_range


def false_alarm_ratio(y, y_pred, min_thresh, max_thresh, interval):

    """ Calculate the False alarm rate across probability thresholds for the top-1 probability. """

    false_alarm_ratios = []
    threshold_range = np.arange(min_thresh, max_thresh + interval, interval)
    for threshold in threshold_range:

        mask = y_pred > threshold
        samples = mask.sum()
        hits = y[mask].sum()
        false_alarms = (samples - hits) / samples
        false_alarm_ratios.append(false_alarms)

    return false_alarm_ratios, threshold_range

def plot_leak_rate_cm(truth, preds, leak_min=0, leak_max=50, interval=5, normalization=None, savefig=False,
                      savepath="./"):
    """
    Plot a confusion matrix of leak rate predictions binned by specified ranges.
    Args:
        truth (np.array): Actual leak rate predictions
        preds (np.array): Predicted Leak rates
        leak_min (int): Minimum leak rate to include in dataset (exclusive)
        leak_max (int): Maximum leak rate to include in dataset (inclusive)
        interval (int): Interval length to bin predictions by
        normalization: Normalization type. Accepts None, "pred", or "true"
        savefig (bool): weather to save the figure out
        savepath (str): Path to save figure
    """
    indices = np.argwhere((truth > leak_min) & (truth <= leak_max))
    leak_rate = truth[indices]
    preds = preds[indices]
    cat_leak_true = np.zeros(shape=leak_rate.squeeze().shape)
    cat_leak_preds = np.zeros(shape=leak_rate.squeeze().shape)
    for i, x in enumerate(range(leak_min, leak_max, interval)):
        true_indices = np.argwhere((leak_rate > x) & (leak_rate <= x + interval))
        pred_indices = np.argwhere((preds.squeeze() > x) & (preds.squeeze() <= x + interval))
        cat_leak_true[true_indices] = i
        cat_leak_preds[pred_indices] = i

    fig, axes = plt.subplots(1, figsize=(10, 10), layout="constrained")
    cm = confusion_matrix(cat_leak_true, cat_leak_preds, normalize=normalization)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{x} > LR <= {x + interval}" for x in
                                                                       range(leak_min, leak_max, interval)])
    disp.plot(xticks_rotation="vertical", ax=axes)
    axes.set_title(f"Leak Rate Confusion Matrix -- Normalization: {str(normalization)}")
    if savefig:
        plt.savefig(f"{savepath}LR_confusion_matrix_normalization_{normalization}_{leak_min}_{leak_max}_{interval}.png",
                    dpi=300, bbox_inches="tight")
    plt.show()
    return disp
