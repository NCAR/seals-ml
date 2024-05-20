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

`mamba env create -f environment_gpu.yml`

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
      sensor_height_min (float): The minimum height (meters) at which the sensors are sampled from
      sensor_height_max (float): The maximum height (meters) at which the sensors are sampled from
      leak_height_min (float): The minimum height (meters) at which the potential leaks are sampled from
      leak_height_max (float): The maximum height (meters) at which the potential leaks are sampled from 
      sensor_type_mask (int): The value to use for the "variable mask" (which sensors are included at specifc locations)
      sensor_exist_mask (int): The value of the "sensor pad mask" (how many sensors there are per sample)
      coord_vars (list): The list of variable names for positional coordinates
      met_vars (list): Names of the meteorological variables 
      emission_vars (list): Variable names of the leak contaminates 

This will create a separate xarray dataset for each file that can be used for model training:

    <xarray.Dataset>
    Dimensions:        (variable: 8, sample: 3600, sensor: 15, time: 20, mask: 2,
                        pot_leak: 10, target_time: 1, sensor_loc: 3)
    Coordinates:
      * variable       (variable) object 'ref_distance' 'ref_azi_sin' ... 'q_CH4'
      * sensor_loc     (sensor_loc) 'xPos' 'yPos' 'zPos'
    Dimensions without coordinates: sample, sensor, time, mask, pot_leak,
                                    target_time
    Data variables:
        encoder_input  (sample, sensor, time, variable, mask) float64 ...
        decoder_input  (sample, pot_leak, target_time, variable, mask) float64 ...
        target         (sample, pot_leak, target_time) float64 ...
        target_ch4     (sample, pot_leak, target_time) float32 ...
        sensor_meta    (sample, sensor, sensor_loc)    float32 ...
        leak_meta      (sample, pot_leak, sensor_loc)  float32 ...
        met_sensor_loc (sample, sensor_loc)            float32 ...
        leak_rate      (sample)                        float32 ...

The `variable` coordinate lists the order of the actual variable names for the `variable` dimension. 
* `ref_distance`: Euclidean distance from the **met sensor** to the sensor / pot_leak location.
* `ref_azi_sin`: Sin of the angle from the **met sensor** relative to the mean wind direction and the sensor / leak loc.
* `ref_azi_cos`: Cosine of the angle from the **met sensor** relative to the mean wind direction and the sensor / leak loc.
* `ref_elv`: Vertical angle from the met sensor and the sensor / leak location.
* `u`: U-wind component at the met sensor.
* `v`: V-wind component at the met sensor.
* `w`: W-wind component at the met sensor.
* `q_CH4`: Methane concentration at a given sensor.

The dimension size of `sensor` and `pot_leak` are the **maximum** number set within the config even though many samples have less than 
the maximum amount. The `mask` dimension identifies which samples have which sensors / potential leaks are contained 
within a given sample.

`Data Variables`:
* `encoder_input`: Time series of meteorlogical and methane inputs as well as coordinate information.
* `decoder_input`: Coordinate information of the potential leak locations. 
* `target`: Binary vector indicating the true leak location from all potential leak locations.
* `target_ch4`: True leak rate at the true vector position.
* `leak_meta`: Raw coordinate information (in meters, directly from LES) for potential leak locations.
* `met_sensor_loc`: Raw coordinate information (in meters, directly from LES) for met sensor.
* `leak_rate`: True leak rate.

### Model Training

To train a transformer model we can use the `./scripts/submit_train.sh` along with the `./config/train_transformer.yaml`.

Explanation of `./config/train_transformer.yaml`

    data_path (str): Path to the generated sample data (created above)
    out_path (str): Path to save output
    random_seed (int): Random seed used to replicate data splitting and model initialization
    save_model (bool): Choice to save output (model, scaler)
    save_output (bool): Choice to save model predictions
    validation_ratio (float): Ratio of validation data compared to training data (between 0 and 1)
    sensor_type_mask (int): Value that was used to mask sensor type (from above config) 
    sensor_exist_mask (int): Value that was used to mask sensor padding (from above config) 
    scaler_type (str): Type of scaler to use (supports "quantile", "standard", "minmax")
    models (list): List of models to train (supports "transformer_leak_loc", "transformer_leak_rate", "backtracker")

    <model type>: Model hyperparameters 
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

      fit:
        epochs (int): Number of epochs to run model
        verbose (int): Verbosity level for training (0, 1, 2)
        batch_size (int): Training batch size
      compile:
        loss (str): Loss function (supported by keras)
        optimizer (str): Optimizer algorithm (supports "adam", "sgd")
        metrics (list): List of training metrics to tracks (supported by keras)

    predict_batch_size (int): Batchsize to use for inference 

