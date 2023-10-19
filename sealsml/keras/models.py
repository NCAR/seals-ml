import keras_core.layers as layers
import keras_core.models as models
from keras_nlp.layers import TransformerDecoder, TransformerEncoder
from .layers import VectorQuantizer


class QuantizedTransformer(models.Model):
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
                 use_quantizer=True,
                 quantized_beta=0.25,
                 n_outputs=1, **kwargs):
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
        self.vector_quantizer = None
        self.n_outputs = n_outputs
        self.encoder_hidden = layers.Dense(self.hidden_size, activation=self.hidden_activation)
        self.decoder_hidden = layers.Dense(self.hidden_size, activation=self.hidden_activation)
        self.transformer_encoder = TransformerEncoder(intermediate_dim=self.hidden_size,
                                                      num_heads=self.n_heads,
                                                      dropout=self.dropout_rate,
                                                      activation=self.hidden_activation)
        self.transformer_decoder = TransformerDecoder(intermediate_dim=self.hidden_size,
                                                      num_heads=self.n_heads,
                                                      dropout=self.dropout_rate,
                                                      activation=self.hidden_activation)
        if self.use_quantizer:
            self.vector_quantizer = VectorQuantizer(self.num_quantized_embeddings, self.hidden_size,
                                                beta=self.quantized_beta)
        self.output_layer = layers.Dense(n_outputs, activation=self.output_activation)

        return

    def call(self, inputs, training=False):
        # First inputs element is the encoder input, which would be the sensors.
        encoder_input = inputs[0]
        # Second inputs element is the decoder input, which would be the potential leak locations.
        decoder_input = inputs[1]
        encoder_hidden_out = self.encoder_hidden(encoder_input)
        decoder_hidden_out = self.decoder_hidden(decoder_input)
        encoder_output = self.transformer_encoder(encoder_hidden_out)
        for e in range(1, self.encoder_layers):
            encoder_output = self.transformer_encoder(encoder_output)
        if self.use_quantizer:
            encoder_output = self.vector_quantizer(encoder_output)
        decoder_output = self.transformer_decoder(decoder_hidden_out, encoder_output)
        for d in range(1, self.decoder_layers):
            decoder_output = self.transformer_decoder(decoder_hidden_out, encoder_output)
        output = self.output_layer(decoder_output)
        return output
