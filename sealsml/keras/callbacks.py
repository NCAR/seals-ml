import keras
from .metrics import mean_searched_locations
import numpy as np

class LeakLocRateMetricsCallback(keras.callbacks.Callback):

    def __init__(self, x_val, y_val, batch_size=2048):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_val, batch_size=self.batch_size)
        cat_accuracy = keras.metrics.CategoricalAccuracy()
        cat_accuracy.update_state(self.y_val[0], y_pred[0])
        logs["val_categorical_accuracy"] = cat_accuracy.result().numpy()
        logs["val_mean_searched_locations"] = mean_searched_locations(self.y_val[0], y_pred[0]).numpy()
        logs["val_rmse"] = np.sqrt(keras.metrics.mean_squared_error(self.y_val[1], y_pred[1][:, 0]).numpy())
        logs["val_mae"] = keras.metrics.mean_absolute_error(self.y_val[1], y_pred[1][:, 0]).numpy()

