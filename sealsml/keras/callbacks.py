import keras
from .metrics import mean_searched_locations, mean_error, sharpness
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
        cat_2_accuracy = keras.metrics.TopKCategoricalAccuracy(k=2)
        cat_3_accuracy = keras.metrics.TopKCategoricalAccuracy(k=3)
        cat_2_accuracy.update_state(self.y_val[0], y_pred[0])
        cat_3_accuracy.update_state(self.y_val[0], y_pred[0])
        cat_accuracy.update_state(self.y_val[0], y_pred[0])
        logs["val_binary_crossentropy"] = np.mean(keras.metrics.binary_crossentropy(self.y_val[0], y_pred[0]).numpy())
        logs["val_categorical_crossentropy"] = np.mean(keras.metrics.categorical_crossentropy(self.y_val[0], y_pred[0]).numpy())
        logs["val_categorical_accuracy"] = np.mean(cat_accuracy.result().numpy())
        logs["val_binary_accuracy"] = np.mean(keras.metrics.binary_accuracy(self.y_val[0], y_pred[0]).numpy())
        logs["val_mean_searched_locations"] = mean_searched_locations(self.y_val[0], y_pred[0]).numpy()
        logs["val_rmse"] = np.sqrt(keras.metrics.mean_squared_error(self.y_val[1], y_pred[1][:, 0]).numpy())
        logs["val_mae"] = keras.metrics.mean_absolute_error(self.y_val[1], y_pred[1][:, 0]).numpy()
        logs["val_mean_error"] = mean_error(self.y_val[1], y_pred[1][:, 0]).numpy()
        logs["val_sharpness"] = sharpness(self.y_val[1], y_pred[1][:, 0]).numpy()
        logs["val_top_2_accuarcy"] = np.mean(cat_2_accuracy.result().numpy())
        logs["val_top_3_accuarcy"] = np.mean(cat_3_accuracy.result().numpy())

class LeakLocMetricsCallback(keras.callbacks.Callback):

    def __init__(self, x_val, y_val, batch_size=1024):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_val, batch_size=self.batch_size)
        cat_accuracy = keras.metrics.CategoricalAccuracy()
        cat_2_accuracy = keras.metrics.TopKCategoricalAccuracy(k=2)
        cat_3_accuracy = keras.metrics.TopKCategoricalAccuracy(k=3)
        cat_accuracy.update_state(self.y_val, y_pred)
        cat_2_accuracy.update_state(self.y_val, y_pred)
        cat_3_accuracy.update_state(self.y_val, y_pred)
        logs["val_binary_crossentropy"] = np.mean(keras.metrics.binary_crossentropy(self.y_val, y_pred).numpy())
        logs["val_categorical_crossentropy"] = np.mean(keras.metrics.categorical_crossentropy(self.y_val, y_pred).numpy())
        logs["val_categorical_accuracy"] = np.mean(cat_accuracy.result().numpy())
        logs["val_binary_accuracy"] = np.mean(keras.metrics.binary_accuracy(self.y_val, y_pred).numpy())
        logs["val_mean_searched_locations"] = mean_searched_locations(self.y_val, y_pred).numpy()
        logs["val_top_2_accuarcy"] = np.mean(cat_2_accuracy.result().numpy())
        logs["val_top_3_accuarcy"] = np.mean(cat_3_accuracy.result().numpy())




