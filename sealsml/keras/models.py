import keras_core.layers as layers
import keras_core.models as models
from keras_nlp.layers import TransformerDecoder, TransformerEncoder
from .layers import VectorQuantizer


class QuantizedTransformer(models.Model):
    def __init__(self, encoder_layers=1, decoder_layers=1,
                 num_quantized_embeddings=500,
                 hidden_size=128,
                 n_heads=8,
                 hidden_activation="relu",
                 output_activation="sigmoid",
                 dropout_rate=0.1,
                 quantized_beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_quantized_embeddings = num_quantized_embeddings
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.hidden_activation = hidden_activation
        self.dropout_rate = dropout_rate
        self.quantized_beta = quantized_beta
        self.output_activation = output_activation
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
        self.vector_quantizer = VectorQuantizer(self.num_quantized_embeddings, self.hidden_size,
                                                beta=self.quantized_beta)
        self.output_layer = layers.Dense(1, activation=self.output_activation)

        return

    def call(self, inputs, training=False):
        encoder_input = inputs[0]
        decoder_input = inputs[1]
        encoder_hidden_out = self.encoder_hidden(encoder_input)
        decoder_hidden_out = self.decoder_hidden(decoder_input)
        encoder_output = self.transformer_encoder(encoder_hidden_out)
        for e in range(1, self.encoder_layers):
            encoder_output = self.transformer_encoder(encoder_output)
        encoder_output = self.vector_quantizer(encoder_output)
        decoder_output = self.transformer_decoder(decoder_hidden_out, encoder_output)
        for d in range(1, self.decoder_layers):
            decoder_output = self.transformer_decoder(decoder_hidden_out, encoder_output)
        output = self.output_layer(decoder_output)
        return output