import keras.layers as layers
import keras.initializers as initializers
import keras_core.ops as ops
import keras

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
