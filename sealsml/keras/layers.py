import keras.layers as layers
import keras.initializers as initializers
import keras.ops as ops
import keras
import keras_nlp
import numpy as np


@keras.saving.register_keras_serializable(package="SEALS_keras")
class VectorQuantizer(layers.Layer):

    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta
        self.embeddings = self.add_weight(
            shape=(self.embedding_dim, self.num_embeddings),
            initializer=initializers.RandomUniform,
            trainable=True,
            name="embeddings_vqvae")

    def call(self, x):
        input_shape = ops.shape(x)
        outputs = []
        for i in range(input_shape[1]):
        # Quantization.
            encoding_indices = self.get_code_indices(x[:, i])
            encodings = ops.one_hot(encoding_indices, self.num_embeddings)
            quantized = ops.matmul(encodings, ops.transpose(self.embeddings))

            # Calculate vector quantization loss and add that to the layer. You can learn more
            # about adding losses to different layers here:
            # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
            # the original paper to get a handle on the formulation of the loss function.
            commitment_loss = ops.mean((ops.stop_gradient(quantized) - x[:, i]) ** 2)
            codebook_loss = ops.mean((quantized - ops.stop_gradient(x[:, i])) ** 2)
            self.add_loss(self.beta * commitment_loss + codebook_loss)

            # Straight-through estimator.
            quantized = x[:, i] + ops.stop_gradient(quantized - x[:, i])
            outputs.append(quantized)
        output = ops.stack(outputs, axis=1)
        return output

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = ops.matmul(flattened_inputs, self.embeddings)
        distances = (
            ops.sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + ops.sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        # Derive the indices for minimum distances.
        encoding_indices = ops.argmin(distances, axis=1)
        return encoding_indices

    def get_config(self):
        base_config = super().get_config()
        param_config = dict(num_embeddings=self.num_embeddings,
                            embedding_dim=self.embedding_dim,
                            beta=self.beta)
        return {**base_config, **param_config}


@keras.saving.register_keras_serializable(package="SEALS_keras")
class ConvSensorEncoder(layers.Layer):
    """
    Performs a series of the same 1D convolutions and poolings on each sensor in an arbitrary length sequence of
    sensors. Expects input of shape (batch_size, sensor, time, variable).

    """
    def __init__(self, min_filters=4, kernel_size=3, filter_growth_rate=2, n_conv_layers=3, hidden_activation="relu",
                 pooling="average", pool_size=2, padding="valid",
                 **kwargs):
        super().__init__(**kwargs)
        self.min_filters = min_filters
        self.kernel_size = kernel_size
        self.filter_growth_rate = filter_growth_rate
        self.n_conv_layers = n_conv_layers
        self.hidden_activation = hidden_activation
        self.pooling = pooling
        self.pool_size = pool_size
        self.padding = padding
        self.conv_layers = []
        self.pooling_layers = []
        curr_filters = min_filters
        for c in range(self.n_conv_layers):
            self.conv_layers.append(layers.SeparableConv1D(curr_filters, self.kernel_size, padding=self.padding,
                                                           activation=self.hidden_activation, use_bias=False))
            if pooling == "average":
                self.pooling_layers.append(layers.AveragePooling1D(self.pool_size))
            else:
                self.pooling_layers.append(layers.MaxPooling1D(self.pool_size))
            curr_filters *= filter_growth_rate
        return

    def call(self, x):
        input_shape = ops.shape(x)
        outputs = []
        for s in range(input_shape[1]):
            sensor_output = x[:, s]
            for c in range(self.n_conv_layers):
                conv_output = self.conv_layers[c](sensor_output)
                sensor_output = self.pooling_layers[c](conv_output)
            sensor_output_shape = ops.shape(sensor_output)
            sensor_output_flat = ops.reshape(sensor_output,
                                             (sensor_output_shape[0], sensor_output_shape[1] * sensor_output_shape[2]))
            outputs.append(sensor_output_flat)
        final_output = ops.stack(outputs, axis=1)
        return final_output

    def get_config(self):
        base_config = super().get_config()
        param_config = dict(min_filters=self.min_filters,
                            kernel_size=self.kernel_size,
                            filter_growth_rate=self.filter_growth_rate,
                            n_conv_layers=self.n_conv_layers,
                            hidden_activation=self.hidden_activation,
                            pooling=self.pooling,
                            pool_size=self.pool_size,
                            padding=self.padding)
        return {**base_config, **param_config}


@keras.saving.register_keras_serializable(package="SEALS_keras")
class TimeBlockSensorEncoder(layers.Layer):
    """
    This layer takes the encoder data structure and breaks the sensor fields into blocks of equal time length.
    Separating the data this way enables the attention layers in transformer models to attend across both time
    and sensor locations, which should make it easier for the model to learn cross-correlations between similar
    plume signals at different locations.

    Attributes:
        embedding_size: number of neurons in the for the dense layer that performs vector embedding
        embedding_activation: activation function for the dense layer
        block_size: number of time steps in each time block
        n_coords: number of coordinate variables in the input data.
        pe_max_wavelength: max wavelength setting for the position encoder. Using default of 10000 for now.

    """
    def __init__(self, embedding_size=512, embedding_activation="relu",
                 block_size=10, n_coords=4, pe_max_wavelength=10000, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_activation = embedding_activation
        self.block_size = block_size
        self.n_coords = n_coords
        self.pe_max_wavelength = pe_max_wavelength
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
            sensor_data = x[:, s, :, self.n_coords:]
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
            x_updated = ops.where(mask == 1, x, -999.0)
        else:
            x_updated = x
        return layers.softmax(x_updated, axis=axis)

