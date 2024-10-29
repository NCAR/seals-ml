import keras.layers as layers
import keras.ops as ops
import keras
import keras_nlp
import numpy as np


@keras.saving.register_keras_serializable(package="SEALS_keras")
class TimeBlockSensorEncoder(layers.Layer):
    """
    This layer takes the encoder data structure and breaks the sensor fields into blocks of equal time length.
    Separating the data this way enables the attention layers in transformer models to attend across both time
    and sensor locations, which should make it easier for the model to learn cross-correlations between similar
    plume signals at different locations.

    Attributes:
        embedding_size (int): number of neurons in the for the dense layer that performs vector embedding
        embedding_activation (str): activation function for the dense layer
        block_size (int): number of time steps in each time block
        n_coords (int): number of coordinate variables in the input data.
        pe_max_wavelength (int): max wavelength setting for the position encoder. Using default of 10000 for now.
        data_start_index (int): Starting index of the data. If only a subset of coordinates is used without the
            data being changed, this allows one to make sure only data variables are ingested into the encoder.

    """
    def __init__(self, embedding_size=512, embedding_activation="relu",
                 block_size=10, n_coords=4, pe_max_wavelength=10000, data_start_index=4, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_activation = embedding_activation
        self.block_size = block_size
        self.n_coords = n_coords
        self.pe_max_wavelength = pe_max_wavelength
        self.data_start_index = data_start_index
        self.position_encoder = keras_nlp.layers.SinePositionEncoding(max_wavelength=self.pe_max_wavelength, name="tb_pe")
        self.dense_embedding = layers.Dense(self.embedding_size, activation=embedding_activation, name="tb_embedding")

    def call(self, x):
        input_shape = ops.shape(x)
        if input_shape[2] % self.block_size != 0:
            raise ValueError("Time series length (x.shape[2]) is not divisible by block_size")
        # Loop over each sensor and apply blocking and embedding separately
        sensor_outputs = []
        for s in range(input_shape[1]):
            # Extract coordinate information from last timestep
            sensor_coords = x[:, s, -1, :self.n_coords]
            # Extract coordinates from all time steps
            sensor_data = x[:, s, :, self.data_start_index:]
            # Break time series into equally-sized sub blocks
            time_blocks = np.arange(0, sensor_data.shape[1] + self.block_size, self.block_size)
            outputs = []
            for t, tb in enumerate(time_blocks[:-1]):
                sensor_block = sensor_data[:, time_blocks[t]:time_blocks[t+1]]
                block_shape = ops.shape(sensor_block)
                # Want to flatten time series of sensor values into one long vector
                sensor_block_flat = ops.reshape(sensor_block,
                                                (block_shape[0], block_shape[1] * block_shape[2]))
                # Append spatial coordinates of sensor onto beginning of vector to prevent coord repeats
                combined_block = ops.concatenate([sensor_coords, sensor_block_flat], axis=1)
                # Use dense layer to do  linear transform of vector into latent space of same size of decoder
                embedded_block = self.dense_embedding(combined_block)
                outputs.append(embedded_block)
            # Combine time blocks vectors into one tensor with shape (batch size, n_blocks, embedding size)
            all_blocks = ops.stack(outputs, axis=1)
            # Calculate position encoder so embedding vector includes info about relative time position of each block
            pe_output = self.position_encoder(all_blocks)
            # Add position encoder onto embedding vector
            final_sensor_output = all_blocks + pe_output
            sensor_outputs.append(final_sensor_output)
        # Stack embeddings of all sensors together into tensor shape (batch size, n_blocks * n_sensors, embedding size)
        # This way the transformer layers can do attention across both time blocks and sensor locations
        all_outputs = ops.concatenate(sensor_outputs, axis=1)
        return all_outputs

    def get_config(self):
        base_config = super().get_config()
        param_config = dict(embedding_size=self.embedding_size,
                            embedding_activation=self.embedding_activation,
                            block_size=self.block_size,
                            n_coords=self.n_coords,
                            pe_max_wavelength=self.pe_max_wavelength)
        base_config.update(param_config)
        return base_config


@keras.saving.register_keras_serializable(package="SEALS_keras")
class MaskedSoftmax(layers.Layer):
    """
    Mask out decoder values in softmax calculation by replacing masked values with -999.
    exp(-999)= 0, so exp(x) / sum(exp(x)) should not be influenced by masked potential leak locations.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, mask=None, axis=-1):
        if mask is not None:
            x_updated = ops.where(mask, x, -999.0)
        else:
            x_updated = x
        return keras.activations.softmax(x_updated, axis=axis)

