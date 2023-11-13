import keras.layers as layers
from keras_nlp.layers import TransformerDecoder, TransformerEncoder
from .layers import VectorQuantizer, ConvSensorEncoder
from keras.saving import deserialize_keras_object
import keras


class QuantizedTransformer(keras.models.Model):
    """
    Transformer model with an optional vector quantizer layer in the encoder branch to produce sharper forecasts.

    Parameters:
        encoder_layers (int): Number of passes through the encoder transformer layer in the encoder branch.
        decoder_layers (int): Number of passes through the decoder transformer layer in the decoder branch.
        hidden_size (int): Size of the latent vector for each element of the sequence (token).
        n_heads (int): number of heads for each attention layer.
        num_quantized_embeddings (int): Number of vectors to store in the vector codebook. The vector codebook will have
            a size of (num_quantized_embeddings, hidden_size).
        hidden_activation (str): Choice of activation function for the hidden layers. Default relu.
        output_activation (str): Choice of activation function for the output layer. Default sigmoid.
        dropout_rate (float): Percentage of neurons to randomly set to 0 during train time.
        use_quantizer (bool): Whether or not to use the VectorQuantizer layer in the model.
        quantized_beta (float): Regularizer term for the commitment loss in the VectorQuantizer layer. Should be between
            0.25 and 2.
        n_outputs (int): Number of outputs being predicted.

    """

    def __init__(self, encoder_layers=1, decoder_layers=1,
                 hidden_size=128,
                 n_heads=8,
                 num_quantized_embeddings=500,
                 hidden_activation="relu",
                 output_activation="sigmoid",
                 dropout_rate=0.1,
                 use_quantizer=False,
                 quantized_beta=0.25,
                 n_outputs=1,
                 min_filters=4, kernel_size=3, filter_growth_rate=2, n_conv_layers=3,
                 pooling="average", pool_size=2, padding="valid",
                 **kwargs):
        super().__init__(**kwargs)
        assert encoder_layers > 0, "Should be at least 1 encoder layer"
        self.encoder_layers = encoder_layers
        assert decoder_layers > 0, "Should be at least 1 decoder layer"
        self.decoder_layers = decoder_layers
        assert num_quantized_embeddings > 0, "num_quantized_embeddings should be positive"
        self.num_quantized_embeddings = num_quantized_embeddings
        assert hidden_size > 0, "hidden_size should be positive"
        self.hidden_size = hidden_size
        assert n_heads > 0, "n_heads should be positive"
        self.n_heads = n_heads
        self.hidden_activation = hidden_activation
        assert 0 <= dropout_rate < 1, "dropout rate should be between 0 and 1"
        self.dropout_rate = dropout_rate
        self.quantized_beta = quantized_beta
        self.output_activation = output_activation
        self.use_quantizer = use_quantizer
        self.n_outputs = n_outputs
        self.min_filters= min_filters
        self.kernel_size = kernel_size
        self.filter_growth_rate = filter_growth_rate
        self.n_conv_layers = n_conv_layers
        self.pooling = pooling
        self.pool_size = pool_size
        self.padding = padding
        self.hyperparameters = ["encoder_layers", "decoder_layers", "hidden_size", "n_heads",
                                "num_quantized_embeddings", "hidden_activation", "output_activation",
                                "dropout_rate", "use_quantizer", "quantized_beta", "n_outputs", "min_filters",
                                "kernel_size", "filter_growth_rate", "n_conv_layers", "pooling", "pool_size", "padding"]
        self.conv_encoder = ConvSensorEncoder(min_filters=self.min_filters, kernel_size=self.kernel_size,
                                              filter_growth_rate=self.filter_growth_rate,
                                              n_conv_layers=self.n_conv_layers,
                                              pooling=self.pooling, padding=self.padding,
                                              hidden_activation=self.hidden_activation)
        self.encoder_hidden = layers.Dense(self.hidden_size, activation=self.hidden_activation,
                                           name="encoder_hidden")
        self.decoder_hidden = layers.Dense(self.hidden_size, activation=self.hidden_activation,
                                           name="decoder_hidden")
        self.encoder_transformers = []
        self.decoder_transformers = []
        self.vector_quantizers = {}
        for n in range(self.encoder_layers):
            self.encoder_transformers.append(TransformerEncoder(intermediate_dim=self.hidden_size,
                                                                num_heads=self.n_heads,
                                                                dropout=self.dropout_rate,
                                                                activation=self.hidden_activation,
                                                                name=f"encoder_transformer_{n:02d}"))
        for n in range(self.decoder_layers):
            self.decoder_transformers.append(TransformerDecoder(intermediate_dim=self.hidden_size,
                                                                num_heads=self.n_heads,
                                                                dropout=self.dropout_rate,
                                                                activation=self.hidden_activation,
                                                                name=f"decoder_transformer_{n:02d}"))

        if self.use_quantizer:
            for n in range(1, self.encoder_layers):
                self.vector_quantizers[f"vector_quantizer_{n:02d}"] = VectorQuantizer(self.num_quantized_embeddings,
                                                                                      self.hidden_size,
                                                                                      beta=self.quantized_beta,
                                                                                      name=f"vector_quantizer_{n:02d}")
        self.output_hidden = layers.Dense(self.n_outputs, activation=self.output_activation, name="output_hidden")
        return

    def call(self, inputs, training=False):
        """
        Args:
            inputs (tuple): Inputs should contain at most
                (encoder_input, decoder_input, encoder_padding_mask, decoder_padding_mask) but the mask variables
                are optional. Using this order is required.
            training (bool): if True run the layers in training mode.

        """
        # First inputs element is the encoder input, which would be the sensors.
        encoder_input = inputs[0]
        # Second inputs element is the decoder input, which would be the potential leak locations.
        decoder_input = inputs[1]
        encoder_padding_mask = None
        decoder_padding_mask = None
        if len(inputs) > 2:
            encoder_padding_mask = inputs[2]
        if len(inputs) > 3:
            decoder_padding_mask = inputs[3]
        encoder_conv_out = self.conv_encoder(encoder_input)
        print(encoder_conv_out.shape)
        encoder_hidden_out = self.encoder_hidden(encoder_conv_out)
        decoder_hidden_out = self.decoder_hidden(decoder_input)
        encoder_output = self.encoder_transformers[0](encoder_hidden_out,
                                                      padding_mask=encoder_padding_mask)
        for e in range(1, self.encoder_layers):
            if self.use_quantizer:
                encoder_output = self.vector_quantizers[f"vector_quantizer_{e:02d}"](encoder_output)
            encoder_output = self.encoder_transformers[e](encoder_output,
                                                          padding_mask=encoder_padding_mask)
        decoder_output = self.decoder_transformers[0](decoder_hidden_out, encoder_output,
                                                      encoder_padding_mask=encoder_padding_mask,
                                                      decoder_padding_mask=decoder_padding_mask)
        for d in range(1, self.decoder_layers):
            decoder_output = self.decoder_transformers[d](decoder_output, encoder_output,
                                                          encoder_padding_mask=encoder_padding_mask,
                                                          decoder_padding_mask=decoder_padding_mask)
        output = self.output_hidden(decoder_output)
        return output

    def get_config(self):
        base_config = super().get_config()
        parameter_config = {hp: getattr(self, hp) for hp in self.hyperparameters}
        return {**base_config, **parameter_config}
