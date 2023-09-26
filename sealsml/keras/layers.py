import keras_core.layers as layers
import keras_core.initializers as initializers
import keras_core.ops as ops


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        self.embeddings = self.add_weight(
            shape=(self.embedding_dim, self.num_embeddings),
            initializer=initializers.RandomUniform,
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = ops.shape(x)
        #flattened = ops.reshape(x, [-1, self.embedding_dim])
        outputs = []
        for i in range(input_shape[1]):
        # Quantization.
            encoding_indices = self.get_code_indices(x[:, i])
            encodings = ops.one_hot(encoding_indices, self.num_embeddings)
            quantized = ops.matmul(encodings, ops.transpose(self.embeddings))

            # Reshape the quantized values back to the original input shape
            # quantized = ops.reshape(quantized, input_shape)

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
