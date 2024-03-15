# SEALS
The goal of this repository is to experiment with different machine learning architectures for identifying methane leak locations. 

## Dependencies
  - python=3.10
  - numpy
  - scipy
  - matplotlib
  - xarray
  - netcdf4
  - pandas
  - scikit-learn
  - pyyaml
  - pytest
  - pip
  - jupyter
  - jupyterlab
  - tqdm
  - numba
  - cython
  - metpy
  - dask
  - pip:
    - bridgescaler
    - keras_nlp
    - hagelslag

## Installation

The easiest way to install this package is using conda. After cloning the repository you can install with

`conda env create -f environment.yml`


## Usage

### Data sampling from a virtualized LES dataset

Training data can be sampled from the raw LES data by running the `./scripts/submit_data_gen.sh` script. It uses the 
`./config/generate_training_data.yaml` yaml file for configuration. A brief explanation of the configuration parameters:

    data_path (str): The path containing the raw LES data 
    out_path (str): The path you'd like to save the sampled data to (it will make a directory if not present).
    time_window_size (int): The number of timesteps to compose the input timeseries 
    samples_per_window (int): The number of individual samples (random sensor locations) per time window
    window_stride (int): The index length of which to slide for a new set of samples
    parallel (bool): Whether to use python multiprocessing for parallelization
    n_processors (int): Number of processors to use is parallel is True
    sampler_args:
      min_trace_sensors (int): Minimum number of Methane trace sensors to sample per training sample
      max_trace_sensors (int): Maximum number of Methane trace sensors to sample per training sample
      min_leak_loc (int): Minimum number of potential leak locations to sample per training sample
      max_leak_loc (int): Maximum number of potential leak locations to sample per training sample
      sensor_height (int): The height (index based) at which the sensors are located (currently same for all) 
      leak_height (int): The height (index based) at which the leak locations are located (currently same for all)
      sensor_type_mask (int): The value to use for the "variable mask" (which sensors are included at specifc locations)
      sensor_exist_mask (int): The value of the "sensor pad mask" (how many sensors ther are per sample)
      coord_vars (list): The list of variable names for positional coordinates
      met_vars (list): Names of the meteorological variables 
      emission_vars (list): Variable names of the leak contaminates 

This will create a separate xarray dataset for each file that can be used for model training:

    <xarray.Dataset>
    Dimensions:        (variable: 8, sample: 3600, sensor: 15, time: 20, mask: 2,
                        pot_leak: 10, target_time: 1)
    Coordinates:
      * variable       (variable) object 'ref_distance' 'ref_azi_sin' ... 'q_CH4'
    Dimensions without coordinates: sample, sensor, time, mask, pot_leak,
                                    target_time
    Data variables:
        encoder_input  (sample, sensor, time, variable, mask) float64 ...
        decoder_input  (sample, pot_leak, target_time, variable, mask) float64 ...
        target         (sample, pot_leak, target_time) float64 ...
  

### Model Training

To train a transformer model we can use the `./scripts/submit_train.sh` along with the `./config/train_transformer.yaml`.

Explanation of `./config/train_transformer.yaml`

    data_path (str): Path to the generated sample data (created above)
    out_path (str): Path to save output
    save (bool): Choice to save output (model, scaler, eval)
    validation_ratio (float): Ratio of validation data compared to training data (between 0 and 1)
    sensor_type_mask (int): Value that was used to mask sensor type (from above config) 
    sensor_exist_mask (int): Value that was used to mask sensor padding (from above config) 
    scaler_type (str): Type of scaler to use (supports "quantile", "standard", "minmax")
    
    model: Model hyperparameters 
      encoder_layers (int): Number of encoder blocks
      decoder_layers (int): Number of decoder blocks
      hidden_size (int): Number of hidden neurons per layer 
      n_heads (int): Number of multi-attention heads
      num_quantized_embeddings (int): Embedding size for vector quantizer (if being used)
      hidden_activation (str): Hidden layer activation function
      output_activation (str): Output layer activation function
      dropout_rate (float): Ratio of neurons to drop (float between 0 and 1)
      use_quantizer (bool): Choice to use vector quantizer
      quantized_beta (float): Quantizer parameter (recommended between 0.25 and 2.0)
      n_outputs (int): Number of output nuerons 
      min_filters (int): Number of filters for timeseries convolutions
      kernel_size (int): Number of kernels for time series convolutions 
      filter_growth_rate (float): Rate at which to increase the filter size per layer
      n_conv_layers (int): Number of convolutional layers
      pooling (str): Convolutional pooling strategy
      pool_size (int): Width to apply pooling strategy
      padding (str): Convoluitonal padding strategy 
    model_fit:
      epochs (int): Number of epochs to run model
      verbose (int): Verbosity level for training (0, 1, 2)
      batch_size (int): Training batch size
    model_compile:
      loss (str): Loss function (supported by keras)
      optimizer (str): Optimizer algorithm (supports "adam", "sgd")
      metrics (list): List of training metrics to tracks (supported by keras) 
    predict_batch_size (int): Batchsize to use for inference 

If `save: True`, it will save out a bridgescaler object and an h5 model object. Eventually it will save out 
various performance metrics and probabilities as well, which is still a work in progress.
