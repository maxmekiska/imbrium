# imbrium [![Downloads](https://pepy.tech/badge/imbrium)](https://pepy.tech/project/imbrium) [![PyPi](https://img.shields.io/pypi/v/imbrium.svg?color=blue)](https://pypi.org/project/imbrium/) [![GitHub license](https://img.shields.io/github/license/maxmekiska/Imbrium?color=black)](https://github.com/maxmekiska/Imbrium/blob/main/LICENSE) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/imbrium.svg)](https://pypi.python.org/project/imbrium/)
 
## Status

| Build | Status|
|---|---|
| `MAIN BUILD`  |  ![master](https://github.com/maxmekiska/imbrium/actions/workflows/main.yml/badge.svg?branch=main) |
|  `DEV BUILD`   |  ![development](https://github.com/maxmekiska/imbrium/actions/workflows/main.yml/badge.svg?branch=development) |

## Pip install

```shell
pip install imbrium
```

Standard and Hybrid Deep Learning Multivariate-Multi-Step & Univariate-Multi-Step
Time Series Forecasting.


                      ██╗███╗░░░███╗██████╗░██████╗░██╗██╗░░░██╗███╗░░░███╗
                      ██║████╗░████║██╔══██╗██╔══██╗██║██║░░░██║████╗░████║
                      ██║██╔████╔██║██████╦╝██████╔╝██║██║░░░██║██╔████╔██║
                      ██║██║╚██╔╝██║██╔══██╗██╔══██╗██║██║░░░██║██║╚██╔╝██║
                      ██║██║░╚═╝░██║██████╦╝██║░░██║██║╚██████╔╝██║░╚═╝░██║
                      ╚═╝╚═╝░░░░░╚═╝╚═════╝░╚═╝░░╚═╝╚═╝░╚═════╝░╚═╝░░░░░╚═╝


## Introduction to Imbrium

Imbrium is a deep learning library that specializes in time series forecasting. Its primary objective is to provide a user-friendly repository of deep learning architectures for this purpose. The focus is on simplifying the process of creating and applying these architectures, with the goal of allowing users to create complex architectures without having to build them from scratch. Instead, the emphasis shifts to high-level configuration of the architectures.

## Contributions

The development and improvement of Imbrium is an ongoing process, and contributions from the community are greatly appreciated. New architectures, in particular, can help Imbrium achieve its goals more quickly. While Imbrium is currently based on Keras, it is open to being extended to other machine learning frameworks such as Pytorch.

Recent research in the field of time series forecasting has shown the potential of graph-based neural networks. If you have experience in this area and would like to contribute architectures to Imbrium, your contributions would be most welcomed.

## Hyperparameter Optimization imbrium 1.1.0
<details>
  <summary>Expand</summary>
Starting from version 1.1.0, imbrium will support experimental hyperparamerter optimization for the model layer config and optimizer arguments. The optimization process uses the Optuna library (https://optuna.org/).

### Optimization via the seeker decorator

To leverage Optimization, use the new classes `OptimizePureUni`, `OptimizeHybridUni`, `OptimizePureMulti` and `OptimizeHybridMulti`. These classes implement optimizable model architecture methods:

`OptimizePureUni` & `OptimizePureMulti`:

  - create_fit_mlp
  - create_fit_rnn
  - create_fit_lstm
  - create_fit_cnn
  - create_fit_gru
  - create_fit_birnn
  - create_fit_bilstm
  - create_fit_bigru
  - create_fit_encdec_rnn
  - create_fit_encdec_lstm
  - create_fit_encdec_gru
  - create_fit_encdec_cnn

`OptimizeHybridUni` & `OptimizeHybridMulti`:

  - create_fit_cnnrnn
  - create_fit_cnnlstm
  - create_fit_cnngru
  - create_fit_cnnbirnn
  - create_fit_cnnbilstm
  - create_fit_cnnbigru

#### Example `OptimizePureUni`

```python
from imbrium.predictors.univarpure import OptimizePureUni
from imbrium.utils.optimization import seeker

# initialize optimizable predictor object
predictor = OptimizePureUni(steps_past=5, steps_future=10, data=data, scale='standard')


# use seeker decorator on optimization harness
@seeker(optimizer_range=["adam", "sgd"], 
        layer_config_range= [
            {'layer0': (5, 'relu'), 'layer1': (10,'relu'), 'layer2': (5, 'relu')},
            {'layer0': (2, 'relu'), 'layer1': (5, 'relu'), 'layer2': (2, 'relu')}
        ], 
        optimization_target='minimize', n_trials = 2)
def create_fit_model(predictor: object, *args, **kwargs):
    # use optimizable create_fit_xxx method
    return predictor.create_fit_lstm(*args, **kwargs)


create_fit_model(predictor, loss='mean_squared_error', metrics='mean_squared_error', epochs=2,
                 show_progress=0, validation_split=0.20, monitor='val_loss', patience=2, min_delta=0, verbose=1
                )

predictor.show_performance()
predictor.predict(data.tail(5))
predictor.model_blueprint()
```

#### Example `OptimizeHybridUni`

```python
from imbrium.predictors.univarhybrid import OptimizeHybridUni
from imbrium.utils.optimization import seeker

predictor = OptimizeHybridUni(sub_seq = 2, steps_past = 10, steps_future = 5, data = data, scale = 'maxabs')

@seeker(optimizer_range=["adam", "sgd"], 
        layer_config_range= [
            {'layer0': (8, 1, 'relu'), 'layer1': (4, 1, 'relu'), 'layer2': (2),'layer3': (25, 'relu'), 'layer4': (10, 'relu')},
            {'layer0': (16, 1, 'relu'), 'layer1': (8, 1, 'relu'), 'layer2': (2),'layer3': (55, 'relu'), 'layer4': (10, 'relu')},
            {'layer0': (32, 1, 'relu'), 'layer1': (16, 1, 'relu'), 'layer2': (2),'layer3': (25, 'relu'), 'layer4': (10, 'relu')}
        ], 
        optimization_target='minimize', n_trials = 2)
def create_fit_model(predictor: object, *args, **kwargs):
    return predictor.create_fit_cnnlstm(*args, **kwargs)

create_fit_model(predictor, loss='mean_squared_error', metrics='mean_squared_error', epochs=2,
                 show_progress=0, validation_split=0.20, monitor='val_loss', patience=2, min_delta=0, verbose=1
                )

predictor.show_performance()
predictor.predict(data.tail(10))
predictor.model_blueprint()
```

#### Example `OptimizePureMulti`

```python
predictor = OptimizePureMulti(steps_past =  5, steps_future = 10, data = data, features = ['target', 'target', 'HouseAge', 'AveRooms', 'AveBedrms'], scale = 'normalize')


@seeker(optimizer_range=["adam", "sgd"], 
        layer_config_range= [
            {'layer0': (5, 'relu'), 'layer1': (10,'relu'), 'layer2': (5, 'relu')},
            {'layer0': (2, 'relu'), 'layer1': (5, 'relu'), 'layer2': (2, 'relu')},
            {'layer0': (20, 'relu'), 'layer1': (50, 'relu'), 'layer2': (20, 'sigmoid')}
        ], 
        optimization_target='minimize', n_trials = 3)
def create_fit_model(predictor: object, *args, **kwargs):
    return predictor.create_fit_lstm(*args, **kwargs)

create_fit_model(predictor, loss='mean_squared_error', metrics='mean_squared_error', epochs=2,
                 show_progress=1, validation_split=0.20, monitor='val_loss', patience=2, min_delta=0, verbose=1
                )


predictor.show_performance()
predictor.predict(data[['target', 'HouseAge', 'AveRooms', 'AveBedrms']].tail(5))
predictor.model_blueprint()
```


#### Example `OptimizeHybridMulti`

```python
predictor = OptimizeHybridMulti(sub_seq = 2, steps_past = 10, steps_future = 5, data = data,features = ['target', 'target', 'HouseAge', 'AveRooms', 'AveBedrms'], scale = 'normalize')


@seeker(optimizer_range=["adam", "sgd"], 
        layer_config_range= [
            {'layer0': (8, 1, 'relu'), 'layer1': (4, 1, 'relu'), 'layer2': (2), 'layer3': (5, 'relu'), 'layer4': (5, 'relu')},
            {'layer0': (8, 1, 'relu'), 'layer1': (4, 1, 'relu'), 'layer2': (2), 'layer3': (5, 'relu'), 'layer4': (5, 'relu')},
            {'layer0': (8, 1, 'relu'), 'layer1': (4, 1, 'relu'), 'layer2': (2), 'layer3': (5, 'relu'), 'layer4': (5, 'relu')}
        ], 
        optimization_target='minimize', n_trials = 3)
def create_fit_model(predictor: object, *args, **kwargs):
    return predictor.create_fit_cnnlstm(*args, **kwargs)

create_fit_model(predictor, loss='mean_squared_error', metrics='mean_squared_error', epochs=2,
                 show_progress=1, validation_split=0.20, monitor='val_loss', patience=2, min_delta=0, verbose=1
                )


predictor.show_performance()
predictor.predict(data[['target', 'HouseAge', 'AveRooms', 'AveBedrms']].tail(10))
predictor.model_blueprint()
```
#### The shell of the seeker harness
  
```python
predictor = OptimizePureMulti(...)

@seeker(optimizer_range=[...], 
        layer_config_range= [
            {...},
            {...},
            {...}
        ], 
        ...)
def create_fit_model(predictor: object, *args, **kwargs): # seeker harness
    return predictor.create_fit_xxx(*args, **kwargs)

create_fit_model(...) # execute seeker harness


predictor.show_performance()
predictor.predict(...)
predictor.model_blueprint()
```


</details>


## imbrium 1.0.0 changes
<details>
  <summary>Expand</summary>

The following important name changes have been performed:

```
- univarstandard => univarpure
- BasicMultStepUniVar => PureUni
- univarhybrid => univarhybrid (unchanged)
- HybridMultStepUniVar => HybridUni
- multivarstandard => multivarpure
- BasicMultSTepMultVar => PureMulti
- multivarhybrid => multivarhybrid (unchanged)
- HybridMultStepMultVar => HybridMulti
```
</details>

## Try imbrium
<details>
  <summary>Expand</summary>
Please ignore all cudart dlerror/warnings, since no GPU is setup in this jupyter binder environment:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxmekiska/ImbriumTesting-Demo/main?labpath=TestImbrium.ipynb) <br>


For more testing, please visit the dedicated Demo & Testing repository at: https://github.com/maxmekiska/ImbriumTesting-Demo

</details>

## Overview of Imbrium's Functionality

<details>
  <summary>Expand</summary>

Imbrium is designed to simplify the application of deep learning models for time series forecasting. The library offers a variety of pre-built architectures, each with a fixed number of layers. However, the user retains full control over the configuration of each layer, including the number of neurons, the type of activation function, loss function, optimizer, and metrics applied. This allows for the flexibility to adapt the architecture to the specific needs of the forecast task at hand. Imbrium also offers a user-friendly interface for training and evaluating these models, making it easy to quickly iterate and test different configurations.


The library differentiates between two
modes:

1. Univariate-Multistep forecasting
2. Multivariate-Multistep forecasting

These two main modes are further divided based on the complexity of the underlying model architectures:

1. Pure
2. Hybrid

Pure supports the following architectures:

- Multilayer perceptron (MLP)
- Recurrent neural network (RNN)
- Long short-term memory (LSTM)
- Gated recurrent unit (GRU)
- Convolutional neural network (CNN)
- Bidirectional recurrent neural network (BI-RNN)
- Bidirectional long-short term memory (BI-LSTM)
- Bidirectional gated recurrent unit (BI-GRU)
- Encoder-Decoder recurrent neural network
- Encoder-Decoder long-short term memory
- Encoder-Decoder convolutional neural network (Encoding via CNN, Decoding via GRU)
- Encoder-Decoder gated recurrent unit

Hybrid supports:

- Convolutional neural network + recurrent neural network (CNN-RNN)
- Convolutional neural network + Long short-term memory (CNN-LSTM)
- Convolutional neural network + Gated recurrent unit (CNN-GRU)
- Convolutional neural network + Bidirectional recurrent neural network (CNN-BI-RNN)
- Convolutional neural network + Bidirectional long-short term memory (CNN-BI-LSTM)
- Convolutional neural network + Bidirectional gated recurrent unit (CNN-BI-GRU)

Please note that each model is supported by a prior input data pre-processing procedure which allows to set a look-back period, look-forward period, sub-sequences division (only for hybrid architectures) and data scaling method.

The following scikit-learn scaling procedures are supported:

- StandardScaler
- MinMaxScaler
- MaxAbsScaler
- Normalizing ([0, 1])
- None (raw data input)

During training/fitting, callback conditions can be defined to guard against
overfitting.

Trained models can furthermore be saved or loaded if the user wishes to do so.

</details>

## How to use imbrium?

<details>
  <summary>Expand</summary>

Attention: Typing has been left in the below examples to ease the configuration readability.

### Univariate Models:

1. Univariate-Multistep forecasting - Pure architectures

```python3
from imbrium.predictors.univarpure import PureUni

predictor = PureUni(steps_past: int, steps_future: int, data = DataFrame(),
                    scale: str = '')

# Choose between one of the architectures:

predictor.create_mlp(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                     metrics: str = 'mean_squared_error',
                     layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (25,'relu'),
                                          'layer2': (25, 'relu')})

predictor.create_rnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                     metrics: str = 'mean_squared_error',
                     layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'),
                                           'layer2': (50, 'relu')})

predictor.create_lstm(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                      metrics: str = 'mean_squared_error',
                      layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'),
                                            'layer2': (50, 'relu')})

predictor.create_gru(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                     metrics: str = 'mean_squared_error',
                     layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'),
                                           'layer2': (50, 'relu')})

predictor.create_cnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                     metrics: str = 'mean_squared_error',
                     layer_config: dict = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu')})

predictor.create_birnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                       metrics: str = 'mean_squared_error',
                       layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})

predictor.create_bilstm(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                        metrics: str = 'mean_squared_error',
                        layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})

predictor.create_bigru(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                       metrics: str = 'mean_squared_error',
                       layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})

predictor.create_encdec_rnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                            metrics: str = 'mean_squared_error',
                            layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})

predictor.create_encdec_lstm(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                             metrics: str = 'mean_squared_error',
                             layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})

predictor.create_encdec_cnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                            metrics: str = 'mean_squared_error',
                            layer_config: dict = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (100, 'relu')})

predictor.create_encdec_gru(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                            metrics: str = 'mean_squared_error',
                            layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})

# Fit the predictor object - more callback settings at:

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping

predictor.fit_model(epochs: int, show_progress: int = 1, validation_split: float = 0.20,
                      monitor='loss', patience=3)

# Have a look at the model performance
predictor.show_performance()

# Make a prediction based on new unseen data
predictor.predict(data: array)

# Safe your model:
predictor.save_model()

# Load a model:
# Step 1: initialize a new predictor object with same characteristics as model to load
# Step 2: Do not pass in any data
# Step 3: Invoke the method load_model()
# optional Step 4: Use the setter method set_model_id(name: str) to give model a name

loading_predictor = PureUni(steps_past: int, steps_future: int)
loading_predictor.load_model(location: str)
loading_predictor.set_model_id(name: str)
```

2. Univariate-Multistep forecasting - Hybrid architectures

```python3
from imbrium.predictors.univarhybrid import HybridUni

predictor = HybridUni(sub_seq: int, steps_past: int, steps_future: int, data =          DataFrame(), scale: str = '')

# Choose between one of the architectures:

predictor.create_cnnrnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                        metrics: str = 'mean_squared_error',
                        layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

predictor.create_cnnlstm(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                         metrics: str = 'mean_squared_error',
                         layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

predictor.create_cnngru(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                        metrics: str = 'mean_squared_error',
                        layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

predictor.create_cnnbirnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                          metrics: str = 'mean_squared_error',
                          layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

predictor.create_cnnbilstm(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                           metrics: str = 'mean_squared_error',
                           layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

predictor.create_cnnbigru(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                          metrics: str = 'mean_squared_error',
                          layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

# Fit the predictor object - more callback settings at:

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping

predictor.fit_model(epochs: int, show_progress: int = 1, validation_split: float = 0.20,
                      monitor='loss', patience=3)

# Have a look at the model performance
predictor.show_performance()

# Make a prediction based on new unseen data
predictor.predict(data: array)

# Safe your model:
predictor.save_model()

# Load a model:
# Step 1: initialize a new predictor object with same characteristics as model to load
# Step 2: Do not pass in any data
# Step 3: Invoke the method load_model()
# optional Step 4: Use the setter method set_model_id(name: str) to give model a name

loading_predictor =  HybridUni(sub_seq: int, steps_past: int, steps_future: int)
loading_predictor.load_model(location: str)
loading_predictor.set_model_id(name: str)
```

### Multivariate Models:

1. Multivariate-Multistep forecasting - Pure architectures

```python3
from imbrium.predictors.multivarpure import PureMulti

# please make sure that the target feature is the first variable in the feature list
predictor = PureMulti(steps_past: int, steps_future: int, data = DataFrame(), features = [], scale: str = '')

# Choose between one of the architectures:

predictor.create_mlp(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                     metrics: str = 'mean_squared_error',
                     layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (25,'relu'),
                                           'layer2': (25, 'relu')})

predictor.create_rnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                     metrics: str = 'mean_squared_error',
                     layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'),
                                           'layer2': (50, 'relu')})

predictor.create_lstm(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                      metrics: str = 'mean_squared_error',
                      layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'),
                                            'layer2': (50, 'relu')})

predictor.create_gru(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                     metrics: str = 'mean_squared_error',
                     layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'),
                                           'layer2': (50, 'relu')})

predictor.create_cnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                     metrics: str = 'mean_squared_error',
                     layer_config: dict = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu')})

predictor.create_birnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                       metrics: str = 'mean_squared_error',
                       layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})

predictor.create_bilstm(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                        metrics: str = 'mean_squared_error',
                        layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})

predictor.create_bigru(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                       metrics: str = 'mean_squared_error',
                       layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})

predictor.create_encdec_rnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                            metrics: str = 'mean_squared_error',
                            layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})

predictor.create_encdec_lstm(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                             metrics: str = 'mean_squared_error',
                             layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})

predictor.create_encdec_cnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                            metrics: str = 'mean_squared_error',
                            layer_config: dict = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (100, 'relu')})

predictor.create_encdec_gru(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                            metrics: str = 'mean_squared_error',
                            layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})

# Fit the predictor object - more callback settings at:

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping

predictor.fit_model(epochs: int, show_progress: int = 1, validation_split: float = 0.20,
                      monitor='loss', patience=3)

# Have a look at the model performance
predictor.show_performance()

# Make a prediction based on new unseen data
predictor.predict(data: array)

# Safe your model:
predictor.save_model()

# Load a model:
# Step 1: initialize a new predictor object with same characteristics as model to load
# Step 2: Do not pass in any data
# Step 3: Invoke the method load_model()
# optional Step 4: Use the setter method set_model_id(name: str) to give model a name

loading_predictor = PureMulti(steps_past: int, steps_future: int)
loading_predictor.load_model(location: str)
loading_predictor.set_model_id(name: str)
```
2. Multivariate-Multistep forecasting - Hybrid architectures

```python3
from imbrium.predictors.multivarhybrid import HybridMulti

# please make sure that the target feature is the first variable in the feature list
predictor = HybridMulti(sub_seq: int, steps_past: int, steps_future: int, data = DataFrame(), features:list = [], scale: str = '')

# Choose between one of the architectures:

predictor.create_cnnrnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                        metrics: str = 'mean_squared_error',
                        layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

predictor.create_cnnlstm(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                         metrics: str = 'mean_squared_error',
                         layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

predictor.create_cnngru(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                        metrics: str = 'mean_squared_error',
                        layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

predictor.create_cnnbirnn(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                          metrics: str = 'mean_squared_error',
                          layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

predictor.create_cnnbilstm(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                           metrics: str = 'mean_squared_error',
                           layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

predictor.create_cnnbigru(optimizer: str = 'adam', loss: str = 'mean_squared_error',
                          metrics: str = 'mean_squared_error',
                          layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

# Fit the predictor object - more callback settings at:

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping

predictor.fit_model(epochs: int, show_progress: int = 1, validation_split: float = 0.20,
                      monitor='loss', patience=3)

# Have a look at the model performance
predictor.show_performance()

# Make a prediction based on new unseen data
predictor.predict(data: array)

# Safe your model:
predictor.save_model()

# Load a model:
# Step 1: initialize a new predictor object with same characteristics as model to load
# Step 2: Do not pass in any data
# Step 3: Invoke the method load_model()
# optional Step 4: Use the setter method set_model_id(name: str) to give model a name

loading_predictor =  HybridMulti(sub_seq: int, steps_past: int, steps_future: int)
loading_predictor.load_model(location: str)
loading_predictor.set_model_id(name: str)
```
</details>

## References

<details>
  <summary>Expand</summary>

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