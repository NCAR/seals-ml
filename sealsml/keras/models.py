import numpy as np
import keras.layers as layers
import keras.regularizers as regularizers
import keras_nlp.layers as nlp_layers
from .layers import TimeBlockSensorEncoder, MaskedSoftmax
import keras
import keras.ops as ops

# If using TensorFlow, this will make GPU ops as deterministic as possible,
# but it will affect the overall performance, so be mindful of that.
# tf.config.experimental.enable_op_determinism()


class BlockTransformer(keras.models.Model):
    """
    Transformer model that can attend across both time blocks and sensors to localize potential
    leaks. The BlockTransformer first converts the time series of each sensor's data along with the position
     of the sensor into uniform time blocks with fixed sized vector representations. Then, the transformer
     encoder layers adjust the input representations based on the relationships with other time-sensor blocks.
     Next, the time-sensor blocks are connected with the potential leak location representations through the decoder
     transformer blocks. Each decoder block is paired against both the other leak locations and the sensor
     representations to determine which location best matches with where the leak is coming from. The model then
     returns a probability of each leak location being the true source of the leak.

     The advantage of the BlockTransformer approach is that it can support a flexible number of both sensors and
     potential leak locations. It also does not require the sensors or leak location to be in the same places as long
     as both use the same wind-relative position coordinates. The model can also potentially support using more or
     fewer time blocks than were used in training, but accuracy is likely to diminish.

    Parameters:
        encoder_layers (int): number of encoder transformer layers
        decoder_layers (int): number of decoder transformer layers
        hidden_size (int): number of neurons in latent representation for both encoder and decoder layers
        n_heads (int): number of attention heads
        hidden_activation (str): nonlinear function applied to each dense or transformer layer inside the model
        output_activation (str): nonlinear function for output. Suggest softmax or sigmoid.
        dropout_rate (float): Rate at which neurons are randomly dropped out in the transformer layers.
        n_outputs (int): number of outputs per potential leak location.
        block_size (int): number of time steps in each block. Will error if block_size is not divisible by the time dimension.
        n_coords (int): number of input variables used for coordinate values.
        data_start_index (int): index of the first data variable. Can be used to help exclude coords or other inputs
            without reprocessing the data.
    """
    def __init__(self, encoder_layers: int = 1,
                 decoder_layers: int = 1,
                 hidden_size: int = 512,
                 n_heads:int = 8,
                 hidden_activation: str = "relu",
                 output_activation: str = "softmax",
                 dropout_rate: float = 0.1,
                 n_outputs: int = 1,
                 block_size: int = 10,
                 n_coords: int = 4,
                 data_start_index: int = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.n_outputs = n_outputs
        self.block_size = block_size
        self.n_coords = n_coords
        self.data_start_index = data_start_index
        self.hyperparameters = ["encoder_layers", "decoder_layers", "hidden_size", "n_heads",
                                "hidden_activation", "output_activation",
                                "dropout_rate", "n_outputs",
                                "block_size", "n_coords"]
        self.time_block_sensor_encoder = TimeBlockSensorEncoder(embedding_size=self.hidden_size,
                                                                block_size=self.block_size,
                                                                n_coords=self.n_coords)
        self.decoder_hidden = layers.Dense(self.hidden_size, activation=self.hidden_activation,
                                           name="decoder_hidden")
        self.encoder_transformers = []
        self.decoder_transformers = []
        self.vector_quantizers = {}
        for n in range(self.encoder_layers):
            self.encoder_transformers.append(nlp_layers.TransformerEncoder(intermediate_dim=self.hidden_size,
                                                                num_heads=self.n_heads,
                                                                dropout=self.dropout_rate,
                                                                activation=self.hidden_activation,
                                                                name=f"encoder_transformer_{n:02d}"))
        for n in range(self.decoder_layers):
            self.decoder_transformers.append(nlp_layers.TransformerDecoder(intermediate_dim=self.hidden_size,
                                                                num_heads=self.n_heads,
                                                                dropout=self.dropout_rate,
                                                                activation=self.hidden_activation,
                                                                name=f"decoder_transformer_{n:02d}"))
        self.output_hidden = layers.Dense(self.n_outputs, name="output_hidden")
        if self.output_activation == "softmax":
            self.output_activation_layer = MaskedSoftmax(name="output_activation_layer")
        else:
            self.output_activation_layer = layers.Activation(self.output_activation, name="output_activation_layer")
        return

    def call(self, inputs: list, training: bool = False):
        """
        Run forward pass of the model in training or inference mode.

        Args:
            inputs: list of input Tensors. Minimally should include [encoder input, decoder input] and can optionally
                include encoder mask and decoder mask.
            training: Whether running in training model or not. Default False.

        Returns:

        """
        # First inputs element is the encoder input, which would be the sensors.
        encoder_input = inputs[0]
        # Second inputs element is the decoder input, which would be the potential leak locations.
        decoder_input = inputs[1][..., :self.n_coords]
        encoder_shape = ops.shape(encoder_input)
        encoder_padding_mask = None
        decoder_padding_mask = None
        if len(inputs) > 2:
            # Repeat the encoder padding mask values for each time block.
            # Output shape should be (batch_size, n_sensors * n_times / block_size )
            encoder_padding_mask = ops.repeat(inputs[2], int(encoder_shape[2] // self.block_size), axis=1)
        if len(inputs) > 3:
            decoder_padding_mask = inputs[3]

        encoder_hidden_out = self.time_block_sensor_encoder(encoder_input)
        decoder_hidden_out = self.decoder_hidden(decoder_input)
        encoder_output = self.encoder_transformers[0](encoder_hidden_out,
                                                      padding_mask=encoder_padding_mask,
                                                      training=training)
        for e in range(1, self.encoder_layers):
            encoder_output = self.encoder_transformers[e](encoder_output,
                                                          padding_mask=encoder_padding_mask,
                                                          training=training)
        decoder_output = self.decoder_transformers[0](decoder_hidden_out, encoder_output,
                                                      encoder_padding_mask=encoder_padding_mask,
                                                      decoder_padding_mask=decoder_padding_mask,
                                                      training=training)
        for d in range(1, self.decoder_layers):
            decoder_output = self.decoder_transformers[d](decoder_output, encoder_output,
                                                          encoder_padding_mask=encoder_padding_mask,
                                                          decoder_padding_mask=decoder_padding_mask,
                                                          training=training)
        output = self.output_hidden(decoder_output)
        output = ops.squeeze(output, axis=-1)
        if self.output_activation == "softmax":
            output = self.output_activation_layer(output, mask=decoder_padding_mask, axis=-1)
            # output = ops.expand_dims(output, -1)
        else:
            output = self.output_activation_layer(output)
        return output

    def get_config(self):
        base_config = super().get_config()
        parameter_config = {hp: getattr(self, hp) for hp in self.hyperparameters}
        return {**base_config, **parameter_config}


class LocalizedLeakRateBlockTransformer(keras.models.Model):
    """
    Transformer model that can attend across both time blocks and sensors to localize potential
    leaks as well as predict the leak rate.

    The LocalizedLeakRateBlockTransformer first converts the time series of each sensor's data along with the position
     of the sensor into uniform time blocks with fixed sized vector representations. Then, the transformer
     encoder layers adjust the input representations based on the relationships with other time-sensor blocks.
     Next, the time-sensor blocks are connected with the potential leak location representations through the decoder
     transformer blocks. Each decoder block is paired against both the other leak locations and the sensor
     representations to determine which location best matches with where the leak is coming from.
     The model first returns a probability of each leak location being the true source of the leak. This probability
     is then used to perform a weighted average of the representations of each leak location. This weighted
     combination is fed to another hidden layer that finally derives an estimate of the leak rate.

    Parameters:
        encoder_layers (int): number of encoder transformer layers
        decoder_layers (int): number of decoder transformer layers
        hidden_size (int): number of neurons in latent representation for both encoder and decoder layers
        n_heads (int): number of attention heads
        hidden_activation (str): nonlinear function applied to each dense or transformer layer inside the model
        output_activation (str): nonlinear function for output. Suggest softmax or sigmoid.
        dropout_rate (float): Rate at which neurons are randomly dropped out in the transformer layers.
        n_outputs (int): number of outputs per potential leak location.
        block_size (int): number of time steps in each block. Will error if block_size is not divisible by the time dimension.
        n_coords (int): number of input variables used for coordinate values.
        data_start_index (int): index of the first data variable. Can be used to help exclude coords or other inputs
            without reprocessing the data.
    """
    def __init__(self,
                 encoder_layers: int = 1,
                 decoder_layers: int = 1,
                 hidden_size: int = 512,
                 n_heads: int =8,
                 hidden_activation: str ="relu",
                 output_activation: str ="softmax",
                 dropout_rate: float = 0.1,
                 n_outputs: int = 1,
                 block_size: int = 10,
                 n_coords: int = 4,
                 data_start_index: int = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.n_outputs = n_outputs
        self.block_size = block_size
        self.n_coords = n_coords
        self.data_start_index = data_start_index
        self.hyperparameters = ["encoder_layers", "decoder_layers", "hidden_size", "n_heads",
                                "hidden_activation", "output_activation",
                                "dropout_rate", "n_outputs",
                                "block_size", "n_coords"]
        self.time_block_sensor_encoder = TimeBlockSensorEncoder(embedding_size=self.hidden_size,
                                                                block_size=self.block_size,
                                                                n_coords=self.n_coords,
                                                                data_start_index=self.data_start_index)
        self.decoder_hidden = layers.Dense(self.hidden_size, activation=self.hidden_activation,
                                           name="decoder_hidden")
        self.encoder_transformers = []
        self.decoder_transformers = []
        self.vector_quantizers = {}
        for n in range(self.encoder_layers):
            self.encoder_transformers.append(nlp_layers.TransformerEncoder(intermediate_dim=self.hidden_size,
                                                                num_heads=self.n_heads,
                                                                dropout=self.dropout_rate,
                                                                activation=self.hidden_activation,
                                                                name=f"encoder_transformer_{n:02d}"))
        for n in range(self.decoder_layers):
            self.decoder_transformers.append(nlp_layers.TransformerDecoder(intermediate_dim=self.hidden_size,
                                                                num_heads=self.n_heads,
                                                                dropout=self.dropout_rate,
                                                                activation=self.hidden_activation,
                                                                name=f"decoder_transformer_{n:02d}"))
        self.output_hidden = layers.Dense(self.n_outputs, name="output_hidden")
        self.leak_rate_hidden = layers.Dense(self.hidden_size, activation=self.hidden_activation, name="rate_hidden")
        self.output_leak_rate = layers.Dense(self.n_outputs, name="rate")
        if self.output_activation == "softmax":
            self.output_activation_layer = MaskedSoftmax(name="location")
        else:
            self.output_activation_layer = layers.Activation(self.output_activation, name="location")

        return

    def call(self, inputs, training=False):
        # First inputs element is the encoder input, which would be the sensors.
        encoder_input = inputs[0]
        # Second inputs element is the decoder input, which would be the potential leak locations.
        decoder_input = inputs[1][..., :self.n_coords]
        encoder_shape = ops.shape(encoder_input)
        encoder_padding_mask = None
        decoder_padding_mask = None
        if len(inputs) > 2:
            # Repeat the encoder padding mask values for each time block.
            # Output shape should be (batch_size, n_sensors * n_times / block_size )
            encoder_padding_mask = ops.repeat(inputs[2], int(encoder_shape[2] // self.block_size), axis=1)
        if len(inputs) > 3:
            decoder_padding_mask = inputs[3]

        encoder_hidden_out = self.time_block_sensor_encoder(encoder_input)
        decoder_hidden_out = self.decoder_hidden(decoder_input)
        encoder_output = self.encoder_transformers[0](encoder_hidden_out,
                                                      padding_mask=encoder_padding_mask,
                                                      training=training)
        for e in range(1, self.encoder_layers):
            encoder_output = self.encoder_transformers[e](encoder_output,
                                                          padding_mask=encoder_padding_mask, 
                                                          training=training)
        decoder_output = self.decoder_transformers[0](decoder_hidden_out, encoder_output,
                                                      encoder_padding_mask=encoder_padding_mask,
                                                      decoder_padding_mask=decoder_padding_mask,
                                                      training=training)
        for d in range(1, self.decoder_layers):
            decoder_output = self.decoder_transformers[d](decoder_output, encoder_output,
                                                          encoder_padding_mask=encoder_padding_mask,
                                                          decoder_padding_mask=decoder_padding_mask,
                                                          training=training)
        output_latent = self.output_hidden(decoder_output)
        output_squeezed = ops.squeeze(output_latent, axis=-1)
        if self.output_activation == "softmax":
            output_location = self.output_activation_layer(output_squeezed, mask=decoder_padding_mask, axis=-1)
            # output = ops.expand_dims(output, -1)
        else:
            output_location = self.output_activation_layer(output_squeezed)
        # Prevent back propgation from the leak rate neural network to minimize scrambling of the localization model.
        output_loc_stop = ops.stop_gradient(output_location)
        decoder_output_stop = ops.stop_gradient(decoder_output)
        # perform a weighted average of the latent representations of the potential leak locations to
        # create a fixed size latent vector localized to the most likely leak location.
        weighted_decoder_hidden = ops.sum(decoder_output_stop * ops.expand_dims(output_loc_stop, axis=-1), axis=1)
        hidden_rate = self.leak_rate_hidden(weighted_decoder_hidden)
        output_rate = self.output_leak_rate(hidden_rate)
        return output_location, output_rate

    def get_config(self):
        base_config = super().get_config()
        parameter_config = {hp: getattr(self, hp) for hp in self.hyperparameters}
        return {**base_config, **parameter_config}


class BlockEncoder(keras.models.Model):
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

    def __init__(self, encoder_layers=1,
                 hidden_size=128,
                 n_heads=8,
                 hidden_activation="relu",
                 output_activation="sigmoid",
                 dropout_rate=0.1,
                 n_outputs=1,
                 block_size=5,
                 n_coords=4,
                 data_start_index=4,
                 **kwargs):
        super().__init__(**kwargs)
        assert encoder_layers > 0, "Should be at least 1 encoder layer"
        self.encoder_layers = encoder_layers
        assert hidden_size > 0, "hidden_size should be positive"
        self.hidden_size = hidden_size
        assert n_heads > 0, "n_heads should be positive"
        self.n_heads = n_heads
        self.hidden_activation = hidden_activation
        assert 0 <= dropout_rate < 1, "dropout rate should be between 0 and 1"
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.n_outputs = n_outputs
        self.block_size = block_size
        self.n_coords = n_coords
        self.data_start_index = data_start_index
        self.hyperparameters = ["encoder_layers", "hidden_size", "n_heads",
                                "hidden_activation", "output_activation", "data_start_index",
                                "dropout_rate", "n_outputs", "block_size", "n_coords"]
        self.kernel_reg = None
        self.time_block_sensor_encoder = TimeBlockSensorEncoder(embedding_size=self.hidden_size,
                                                                block_size=self.block_size,
                                                                n_coords=self.n_coords,
                                                                data_start_index=self.data_start_index)
        self.encoder_hidden = layers.Dense(self.hidden_size, activation=self.hidden_activation,
                                           kernel_regularizer=self.kernel_reg, name="encoder_hidden")
        self.encoder_transformers = []
        for n in range(self.encoder_layers):
            self.encoder_transformers.append(nlp_layers.TransformerEncoder(intermediate_dim=self.hidden_size,
                                                                num_heads=self.n_heads,
                                                                dropout=self.dropout_rate,
                                                                activation=self.hidden_activation,
                                                                name=f"encoder_transformer_{n:02d}"))

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
        print("Encoder input shape:", encoder_input.shape)
        # Second inputs element is the decoder input, which would be the potential leak locations.
        encoder_padding_mask = None
        encoder_shape = ops.shape(encoder_input)
        if len(inputs) > 2:
            # Repeat the encoder padding mask values for each time block.
            # Output shape should be (batch_size, n_sensors * n_times / block_size )
            encoder_padding_mask = ops.repeat(inputs[2], int(encoder_shape[2] // self.block_size), axis=1)
        encoder_conv_out = self.time_block_sensor_encoder(encoder_input)
        encoder_hidden_out = self.encoder_hidden(encoder_conv_out)
        encoder_output = self.encoder_transformers[0](encoder_hidden_out,
                                                      padding_mask=encoder_padding_mask)
        for e in range(1, self.encoder_layers):
            encoder_output = self.encoder_transformers[e](encoder_output,
                                                          padding_mask=encoder_padding_mask)

        encoder_output_flat = ops.reshape(encoder_output,
                                          newshape=(-1, encoder_output.shape[-2] * encoder_output.shape[-1]))
        output = self.output_hidden(encoder_output_flat)

        return output

    def get_config(self):
        base_config = super().get_config()
        parameter_config = {hp: getattr(self, hp) for hp in self.hyperparameters}
        return {**base_config, **parameter_config}


class BackTrackerDNN(keras.models.Model):
    """
    A Dense Neural Network Model that can support arbitrary numbers of hidden layers
    and provides evidential uncertainty estimation.
    Inherits from BaseRegressor.

    Attributes:
        hidden_layers: Number of hidden layers.
        hidden_neurons: Number of neurons in each hidden layer.
        activation: Type of activation function.
        optimizer: Name of optimizer or optimizer object.
        loss: Name of loss function or loss object.
        use_noise: Whether additive Gaussian noise layers are included in the network.
        noise_sd: The standard deviation of the Gaussian noise layers.
        use_dropout: Whether Dropout layers are added to the network.
        dropout_alpha: Proportion of neurons randomly set to 0.
        batch_size: Number of examples per batch.
        epochs: Number of epochs to train.
        verbose: Level of detail to provide during training.
        model: Keras Model object.
        evidential_coef: Evidential regularization coefficient.
        metrics: Optional list of metrics to monitor during training.
    """
    def __init__(self, hidden_layers=3, hidden_neurons=64, activation="relu", output_activation="sigmoid", optimizer="SGD", loss_weights=None,
                 use_noise=False, noise_sd=0.01, lr=0.00001, use_dropout=False, dropout_alpha=0.1, batch_size=1,
                 epochs=10, kernel_reg=None, l1_weight=0.01, l2_weight=0.01, n_output_tasks=4, verbose=1, **kwargs):

        super().__init__(**kwargs)
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.optimizer_obj = None
        self.loss_weights = loss_weights
        self.lr = lr
        self.kernel_reg = kernel_reg
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.batch_size = batch_size
        self.use_noise = use_noise
        self.noise_sd = noise_sd
        self.use_dropout = use_dropout
        self.dropout_alpha = dropout_alpha
        self.epochs = epochs
        self.optimizer_obj = None
        self.n_output_tasks = n_output_tasks
        self.verbose = verbose
        self.N_OUTPUT_PARAMS = 4
        self.hyperparameters = ["hidden_layers", "hidden_neurons", "activation",
                                "optimizer", "loss_weights", "lr", "kernel_reg", "l1_weight", "l2_weight",
                                "batch_size", "use_noise", "noise_sd", "use_dropout", "dropout_alpha", "epochs",
                                "verbose", "n_output_tasks"]

        if self.activation == "leaky":
            self.activation = layers.LeakyReLU()
        if self.kernel_reg == "l1":
            self.kernel_reg = regularizers.L1(self.l1_weight)
        elif self.kernel_reg == "l2":
            self.kernel_reg = regularizers.L2(self.l2_weight)
        elif self.kernel_reg == "l1_l2":
            self.kernel_reg = regularizers.L1L2(self.l1_weight, self.l2_weight)
        else:
            self.kernel_reg = None
        self.model_layers = []
        for h in range(self.hidden_layers-1):
            self.model_layers.append(layers.Dense(self.hidden_neurons,
                                           activation=self.activation,
                                           kernel_regularizer=self.kernel_reg,
                                           name=f"dense_{h:02d}"))
            if self.use_dropout:
                self.model_layers.append(layers.Dropout(self.dropout_alpha, name=f"dropout_{h:02d}"))
            if self.use_noise:
                self.model_layers.append(layers.GaussianNoise(self.noise_sd, name=f"noise_{h:02d}"))

        # For last hidden layer, reduce number of neurons of previous layer by half
        if self.hidden_neurons >= 64:
            n_half=np.int32(self.hidden_neurons/2)
        else:
            n_half=self.hidden_neurons
        self.model_layers.append(layers.Dense(n_half,
                                       activation=self.activation, 
                                       kernel_regularizer=self.kernel_reg,
                                       name=f"dense_{h:02d}"))

        self.model_layers.append(layers.Dense(self.n_output_tasks, activation=self.output_activation, name="dense_output"))

    def call(self, inputs):

        layer_output = self.model_layers[0](inputs)

        for l in range(1, len(self.model_layers)):
            layer_output = self.model_layers[l](layer_output)

        return layer_output

    def fit(self, x, y, validation_data=None, **kwargs):

        if isinstance(validation_data, tuple):
            validation = (validation_data[0][0], validation_data[1])
        else:
            validation = None

        hist = super().fit(x[0], y, batch_size=self.batch_size, epochs=self.epochs,
                       verbose=self.verbose, validation_data=validation)

        return hist

    def predict(self, x, batch_size=1):

        return super().predict(x[0], batch_size=batch_size)

    def get_config(self):

        base_config = super().get_config()
        parameter_config = {hp: getattr(self, hp) for hp in self.hyperparameters}
        return {**base_config, **parameter_config}

class BlockRateEncoder(keras.models.Model):
        """
        Transformer model that can attend across both time blocks and sensors to localize potential
        leaks.

        Parameters:
            encoder_layers (int): number of encoder transformer layers
            decoder_layers (int): number of decoder transformer layers
            hidden_size (int): number of neurons in latent representation for both encoder and decoder layers
            n_heads (int): number of attention heads
            hidden_activation (str): nonlinear function applied to each dense or transformer layer inside the model
            output_activation (str): nonlinear function for output. Suggest softmax or sigmoid.
            dropout_rate (float): Rate at which neurons are randomly dropped out in the transformer layers.
            n_outputs (int): number of outputs per potential leak location.
            block_size (int): number of time steps in each block. Will error if block_size is not divisible by the time dimension.
            n_coords (int): number of input variables used for coordinate values.
            data_start_index (int): index of the first data variable. Can be used to help exclude coords or other inputs
                without reprocessing the data.
        """

        def __init__(self, encoder_layers=1, decoder_layers=1,
                     hidden_size=512,
                     n_heads=8,
                     hidden_activation="relu",
                     output_activation="linear",
                     dropout_rate=0.1,
                     n_outputs=1,
                     block_size=10,
                     n_coords=4,
                     data_start_index=4,
                     **kwargs):
            super().__init__(**kwargs)
            self.encoder_layers = encoder_layers
            self.hidden_size = hidden_size
            self.n_heads = n_heads
            self.hidden_activation = hidden_activation
            self.output_activation = output_activation
            self.dropout_rate = dropout_rate
            self.n_outputs = n_outputs
            self.block_size = block_size
            self.n_coords = n_coords
            self.data_start_index = data_start_index
            self.hyperparameters = ["encoder_layers", "hidden_size", "n_heads",
                                    "hidden_activation", "output_activation",
                                    "dropout_rate", "n_outputs",
                                    "block_size", "n_coords"]
            self.time_block_sensor_encoder = TimeBlockSensorEncoder(embedding_size=self.hidden_size,
                                                                    block_size=self.block_size,
                                                                    n_coords=self.n_coords)
            self.decoder_hidden = layers.Dense(self.hidden_size, activation=self.hidden_activation,
                                               name="decoder_hidden")
            self.encoder_transformers = []
            self.decoder_transformers = []
            self.vector_quantizers = {}
            for n in range(self.encoder_layers):
                self.encoder_transformers.append(nlp_layers.TransformerEncoder(intermediate_dim=self.hidden_size,
                                                                    num_heads=self.n_heads,
                                                                    dropout=self.dropout_rate,
                                                                    activation=self.hidden_activation,
                                                                    name=f"encoder_transformer_{n:02d}"))
            for n in range(self.decoder_layers):
                self.decoder_transformers.append(nlp_layers.TransformerDecoder(intermediate_dim=self.hidden_size,
                                                                    num_heads=self.n_heads,
                                                                    dropout=self.dropout_rate,
                                                                    activation=self.hidden_activation,
                                                                    name=f"decoder_transformer_{n:02d}"))
            self.output_hidden = layers.Dense(self.n_outputs, name="output_hidden")
            if self.output_activation == "softmax":
                self.output_activation_layer = MaskedSoftmax(name="output_activation_layer")
            else:
                self.output_activation_layer = layers.Activation(self.output_activation, name="output_activation_layer")
            return

        def call(self, inputs, training=False):
            # First inputs element is the encoder input, which would be the sensors.
            encoder_input = inputs[0]
            # Second inputs element is the decoder input, which would be the potential leak locations.
            decoder_input = inputs[1][..., :self.n_coords]
            encoder_shape = ops.shape(encoder_input)
            encoder_padding_mask = None
            decoder_padding_mask = None
            if len(inputs) > 2:
                # Repeat the encoder padding mask values for each time block.
                # Output shape should be (batch_size, n_sensors * n_times / block_size )
                encoder_padding_mask = ops.repeat(inputs[2], int(encoder_shape[2] // self.block_size), axis=1)
            if len(inputs) > 3:
                decoder_padding_mask = inputs[3]

            encoder_hidden_out = self.time_block_sensor_encoder(encoder_input)
            decoder_hidden_out = self.decoder_hidden(decoder_input)
            encoder_output = self.encoder_transformers[0](encoder_hidden_out,
                                                          padding_mask=encoder_padding_mask,
                                                          training=training)
            for e in range(1, self.encoder_layers):
                encoder_output = self.encoder_transformers[e](encoder_output,
                                                              padding_mask=encoder_padding_mask,
                                                              training=training)
            decoder_output = self.decoder_transformers[0](decoder_hidden_out, encoder_output,
                                                          encoder_padding_mask=encoder_padding_mask,
                                                          decoder_padding_mask=decoder_padding_mask,
                                                          training=training)
            for d in range(1, self.decoder_layers):
                decoder_output = self.decoder_transformers[d](decoder_output, encoder_output,
                                                              encoder_padding_mask=encoder_padding_mask,
                                                              decoder_padding_mask=decoder_padding_mask,
                                                              training=training)
            output = self.output_hidden(decoder_output)
            output = ops.squeeze(output, axis=-1)
            if self.output_activation == "softmax":
                output = self.output_activation_layer(output, mask=decoder_padding_mask, axis=-1)
                # output = ops.expand_dims(output, -1)
            else:
                output = self.output_activation_layer(output)
            return output

        def get_config(self):
            base_config = super().get_config()
            parameter_config = {hp: getattr(self, hp) for hp in self.hyperparameters}
            return {**base_config, **parameter_config}
