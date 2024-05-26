# imbrium [![Downloads](https://pepy.tech/badge/imbrium)](https://pepy.tech/project/imbrium) [![PyPi](https://img.shields.io/pypi/v/imbrium.svg?color=blue)](https://pypi.org/project/imbrium/) [![GitHub license](https://img.shields.io/github/license/maxmekiska/Imbrium?color=black)](https://github.com/maxmekiska/Imbrium/blob/main/LICENSE) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/imbrium.svg)](https://pypi.python.org/project/imbrium/)
 
## Status

| Build | Status|
|---|---|
| `MAIN BUILD`  |  ![master](https://github.com/maxmekiska/imbrium/actions/workflows/main.yml/badge.svg?branch=main) |
| `DEV BUILD`   |  ![development](https://github.com/maxmekiska/imbrium/actions/workflows/main.yml/badge.svg?branch=development) |

## Pip install

```shell
pip install imbrium
```

Standard and Hybrid Deep Learning Multivariate-Multi-Step & Univariate-Multi-Step
Time Series Forecasting.


                          ██╗███╗░░░███╗██████╗░██████╗░██╗██╗░░░██╗███╗░░░███╗
                            ║████╗░████║██╔══██╗██╔══██╗██║██║░░░██║████╗░████║
                          ██║██╔████╔██║██████╦╝██████╔╝██║██║░░░██║██╔████╔██║
                          ██║██║╚██╔╝██║██╔══██╗██╔══██╗██║██║░░░██║██║╚██╔╝██║
                          ██║██║░╚═╝░██║██████╦╝██║░░██║██║╚██████╔╝██║░╚═╝░██║
                          ╚═╝╚═╝░░░░░╚═╝╚═════╝░╚═╝░░╚═╝╚═╝░╚═════╝░╚═╝░░░░░╚═╝


## Introduction to imbrium

imbrium is a deep learning library that specializes in time series forecasting. Its primary objective is to provide a user-friendly repository of deep learning architectures for this purpose. The focus is on simplifying the process of creating and applying these architectures, with the goal of allowing users to create complex architectures without having to build them from scratch. Instead, the emphasis shifts to high-level configuration of the architectures.


## imbrium Summary

imbrium is designed to simplify the application of deep learning models for time series forecasting. The library offers a variety of pre-built architectures. The user retains full control over the configuration of each layer, including the number of neurons, the type of activation function, loss function, optimizer, and metrics applied. This allows for the flexibility to adapt the architecture to the specific needs of the forecast task at hand. Imbrium also offers a user-friendly interface for training and evaluating these models, making it easy to quickly iterate and test different configurations.

imbrium uses the sliding window approach to generate forecasts. The sliding window approach in time series forecasting involves moving a fixed-size window - `steps_past` through historical data, using the data within the window as input features. The next data points outside the window are used as the target variables - `steps_future`. This method allows the model to learn sequential patterns and trends in the data, enabling accurate predictions for future points in the time series. 


### Get started with imbrium

<details>
  <summary>Expand</summary>
  <br>

Note, if you choose to use Tensor Board, run the follwoing commonad to disply the results:

```shell
 tensorboard --logdir logs/
```

<details>
  <summary>Univariate Pure Predictors</summary>
  <br>


```python
from imbrium import PureUni

# create a PureUni object (numpy array expected)
predictor = PureUni(target = target_numpy_array, evaluation_split = 0.1, validation_split = 0.2) 

# the following models are available for a PureUni objects;

# create and fit a muti-layer perceptron model
predictor.create_fit_mlp( 
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        dense_block_one = 1,
        dense_block_two = 1,
        dense_block_three = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {"neurons": 50, "activation": "relu", "regularization": 0.0}
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a recurrent neural network model
predictor.create_fit_rnn(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        rnn_block_one = 1,
        rnn_block_two = 1,
        rnn_block_three = 1,
        metrics = "mean_squared_error",
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {"neurons": 50, "activation": "relu", "regularization": 0.0}
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a long short-term neural network model
predictor.create_fit_lstm(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        lstm_block_one = 1,
        lstm_block_two = 1,
        lstm_block_three = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {"neurons": 50, "activation": "relu", "regularization": 0.0}
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional neural network
predictor = create_fit_cnn(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        dense_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a gated recurrent unit neural network  
predictor.create_fit_gru(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        gru_block_one = 1,
        gru_block_two = 1,
        gru_block_three = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a bidirectional recurrent neural network
predictor.create_fit_birnn(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        birnn_block_one = 1,
        rnn_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a bidirectional long short-term memory neural network
predictor.create_fit_bilstm(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        bilstm_block_one = 1,
        lstm_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a bidirectional gated recurrent neural network
predictor.create_fit_bigru(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        bigru_block_one = 1,
        gru_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# you can add additional layer to the defualt layers by increasing the layer block count and adding the configuration for the layer in the layer_config dictionary. Please note that the last layer should not have a dropout key.

# in addition, you can add to all models early stopping arguments:

monitor='val_loss',
min_delta=0,
patience=0,
verbose=0,
mode='auto',
baseline=None,
restore_best_weights=False,
start_from_epoch=0


# instpect model structure
predictor.model_blueprint()

# evaluate model performance on testing data
predictor.evaluate_model(board=False)

# get model performance on testing data
predictor.show_evaluation()

# insptect keras model performances via (access dictionary via history key):
predictor.show_performance()

# make predictions via (numpy array expected):
predictor.predict(data)

# save predictor via:
predictor.freeze(absolute_path)

# load saved predictor via:
predictor.retrieve(location)
```  

</details>

<details>
  <summary>Multivariate Pure Predictors</summary>
  <br>


```python
from imbrium import PureMulti

# create a PureMulti object (numpy array expected)
predictor = PureMulti(target = target_numpy_array, features = features_numpy_array, evaluation_split = 0.1, validation_split = 0.2) 

# the following models are available for a PureMulti objects;

# create and fit a muti-layer perceptron model
predictor.create_fit_mlp( 
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        dense_block_one = 1,
        dense_block_two = 1,
        dense_block_three = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {"neurons": 50, "activation": "relu", "regularization": 0.0}
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a recurrent neural network model
predictor.create_fit_rnn(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        rnn_block_one = 1,
        rnn_block_two = 1,
        rnn_block_three = 1,
        metrics = "mean_squared_error",
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {"neurons": 50, "activation": "relu", "regularization": 0.0}
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a long short-term neural network model
predictor.create_fit_lstm(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        lstm_block_one = 1,
        lstm_block_two = 1,
        lstm_block_three = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {"neurons": 50, "activation": "relu", "regularization": 0.0}
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional neural network
predictor = create_fit_cnn(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        dense_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a gated recurrent unit neural network  
predictor.create_fit_gru(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        gru_block_one = 1,
        gru_block_two = 1,
        gru_block_three = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a bidirectional recurrent neural network
predictor.create_fit_birnn(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        birnn_block_one = 1,
        rnn_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a bidirectional long short-term memory neural network
predictor.create_fit_bilstm(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        bilstm_block_one = 1,
        lstm_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a bidirectional gated recurrent neural network
predictor.create_fit_bigru(
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        bigru_block_one = 1,
        gru_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# you can add additional layer to the defualt layers by increasing the layer block count and adding the configuration for the layer in the layer_config dictionary. Please note that the last layer should not have a dropout key.

# in addition, you can add to all models early stopping arguments:

monitor='val_loss',
min_delta=0,
patience=0,
verbose=0,
mode='auto',
baseline=None,
restore_best_weights=False,
start_from_epoch=0


# instpect model structure
predictor.model_blueprint()

# evaluate model performance on testing data
predictor.evaluate_model(board=False)

# get model performance on testing data
predictor.show_evaluation()

# insptect keras model performances via (access dictionary via history key):
predictor.show_performance()

# make predictions via (numpy array expected):
predictor.predict(data)

# save predictor via:
predictor.freeze(absolute_path)

# load saved predictor via:
predictor.retrieve(location)
```  
</details>

<details>
  <summary>Univariate Hybrid Predictors</summary>
  <br>

```python
from imbrium import HybridUni

# create a HybridUni object (numpy array expected)
predictor = HybridUni(target = target_numpy_array, evaluation_split = 0.1, validation_split = 0.2) 

# the following models are available for a HybridUni objects:
# create and fit a convolutional recurrent neural network
predictor.create_fit_cnnrnn(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        rnn_block_one = 1,
        rnn_block_two = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional long short-term memory neural network
predictor.create_fit_cnnlstm(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        lstm_block_one = 1,
        lstm_block_two = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional gated recurrent unit neural network  
predictor.create_fit_cnngru(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        gru_block_one = 1,
        gru_block_two = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional bidirectional recurrent neural network
predictor.create_fit_cnnbirnn(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        birnn_block_one = 1,
        rnn_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional bidirectional long short-term neural network
predictor.create_fit_cnnbilstm(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        bilstm_block_one = 1,
        lstm_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional bidirectional gated recurrent neural network
predictor.create_fit_cnnbigru(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        bigru_block_one = 1,
        gru_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# you can add additional layer to the defualt layers by increasing the layer block count and adding the configuration for the layer in the layer_config dictionary. Please note that the last layer should not have a dropout key.

# in addition, you can add to all models early stopping arguments:

monitor='val_loss',
min_delta=0,
patience=0,
verbose=0,
mode='auto',
baseline=None,
restore_best_weights=False,
start_from_epoch=0


# instpect model structure
predictor.model_blueprint()

# evaluate model performance on testing data
predictor.evaluate_model(board=False)

# get model performance on testing data
predictor.show_evaluation()

# insptect keras model performances via (access dictionary via history key):
predictor.show_performance()

# make predictions via (numpy array expected):
# - when loading/retrieving a saved model, provide sub_seq, steps_past, steps_future in the predict method!
predictor.predict(data, sub_seq=None, steps_past=None, steps_future=None)

# save predictor via:
predictor.freeze(absolute_path)

# load saved predictor via:
predictor.retrieve(location)
```  

</details>

<details>
  <summary>Multivariate Hybrid Predictors</summary>
  <br>


```python
from imbrium import HybridMulti

# create a HybridMulti object (numpy array expected)
predictor = HybridMulti(target = target_numpy_array, features = features_numpy_array, evaluation_split = 0.1, validation_split = 0.2) 

# the following models are available for a HybridMulti objects:
# create and fit a convolutional recurrent neural network
predictor.create_fit_cnnrnn(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        rnn_block_one = 1,
        rnn_block_two = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional long short-term memory neural network
predictor.create_fit_cnnlstm(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        lstm_block_one = 1,
        lstm_block_two = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional gated recurrent unit neural network  
predictor.create_fit_cnngru(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        gru_block_one = 1,
        gru_block_two = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional bidirectional recurrent neural network
predictor.create_fit_cnnbirnn(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        birnn_block_one = 1,
        rnn_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional bidirectional long short-term neural network
predictor.create_fit_cnnbilstm(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        bilstm_block_one = 1,
        lstm_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# create and fit a convolutional bidirectional gated recurrent neural network
predictor.create_fit_cnnbigru(
        sub_seq,
        steps_past,
        steps_future,
        optimizer = "adam",
        optimizer_args = None,
        loss = "mean_squared_error",
        metrics = "mean_squared_error",
        conv_block_one = 1,
        conv_block_two = 1,
        bigru_block_one = 1,
        gru_block_one = 1,
        layer_config = {
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs = 100,
        batch_size = None,
        show_progress = 1,
        board = False,
)

# you can add additional layer to the defualt layers by increasing the layer block count and adding the configuration for the layer in the layer_config dictionary. Please note that the last layer should not have a dropout key.

# in addition, you can add to all models early stopping arguments:

monitor='val_loss',
min_delta=0,
patience=0,
verbose=0,
mode='auto',
baseline=None,
restore_best_weights=False,
start_from_epoch=0


# instpect model structure
predictor.model_blueprint()

# evaluate model performance on testing data
predictor.evaluate_model(board=False)

# get model performance on testing data
predictor.show_evaluation()

# insptect keras model performances via (access dictionary via history key):
predictor.show_performance()

# make predictions via (numpy array expected):
# - when loading/retrieving a saved model, provide sub_seq, steps_past, steps_future in the predict method!
predictor.predict(data, sub_seq=None, steps_past=None, steps_future=None)

# save predictor via:
predictor.freeze(absolute_path)

# load saved predictor via:
predictor.retrieve(location)
```  
</details>

</details>

### Use Case: scaling + hyper parameter optimization

https://github.com/maxmekiska/ImbriumTesting-Demo/blob/main/use-case-1.ipynb

### Integration tests

https://github.com/maxmekiska/ImbriumTesting-Demo/blob/main/IntegrationTest.ipynb


## References

<details>
  <summary>Expand</summary>
  <br>

Brwonlee, J., 2016. Display deep learning model training history in keras [Online]. Available from:
https://machinelearningmastery.com/display-deep-
learning-model-training-history-in-keras/.

Brwonlee, J., 2018a. How to develop convolutional neural network models for time series forecasting [Online]. Available from:
https://machinelearningmastery.com/how-to-develop-convolutional-
neural-network-models-for-time-series-forecasting/.

Brwonlee, J., 2018b. How to develop lstm models for time series forecasting [Online]. Available from:
https://machinelearningmastery.com/how-to-develop-
lstm-models-for-time-series-forecasting/.

Brwonlee, J., 2018c. How to develop multilayer perceptron models for time series forecasting [Online]. Available from:
https://machinelearningmastery.com/how-to-develop-multilayer-
perceptron-models-for-time-series-forecasting/.

</details>


</details>