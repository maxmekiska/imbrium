# Imbrium


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


## Basics

This library aims to ease the application of deep learning models for time
series forecasting. Multiple architectures are offered with a fixed
number of layers however the user has full control over the number of neurons
per layer, activation function type, loss function type, optimizer type and
metrics applied.
The library differentiates between two
modes:

1. Univariate-Multistep forecasting
2. Multivariate-Multistep forecasting

These two main modes are further divided based on the complexity of the underlying model architectures:

1. Standard
2. Hybrid

Standard supports the following architectures:

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

Please note that each model is supported by a prior input data pre-processing procedure which allows to set how many datapoints should look a model look back for a prediction, how many datapoints should be predicted into the future, how many sub-sequences should be considered (only for hybrid architectures) and what scaling should be applied.

The following scikit-learn scaling procedures are supported:

- StandardScaler
- MinMaxScaler
- MaxAbsScaler
- Normalizing ([0, 1])
- None (raw data input)

Trained models can furthermore be saved or loaded if the user wishes to do so.

## How to use Imbrium?

Simplified workflows, more possible.

### Univariate Models:

1. Univariate-Multistep forecasting - Standard architectures

```python3
from imbrium.predictors.univarstandard import BasicMultStepUniVar

predictor = BasicMultStepUniVar(steps_past: int, steps_future: int, data = pd.DataFrame(), scale: str = '')

# Choose between one of the architectures:

# predictor.create_mlp(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (25,'relu'), 'layer2': (25, 'relu')})
# predictor.create_rnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'), 'layer2': (50, 'relu')})
# predictor.create_lstm(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'), 'layer2': (50, 'relu')})
# predictor.create_gru(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'), 'layer2': (50, 'relu')})
# predictor.create_cnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu')})
# predictor.create_birnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})
# predictor.create_bilstm(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})
# predictor.create_bigru(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})
# predictor.create_encdec_rnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})
# predictor.create_encdec_lstm(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})
# predictor.create_encdec_cnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (100, 'relu')})
# predictor.create_encdec_gru(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})

# Fit the predictor object
predictor.fit_model(epochs: int, show_progress: int = 1, validation_split: float = 0.20, batch_size: int = 10)

# Have a look at the model performance - MSE based, more evaluation forms might be added on architecture level in the future
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

loading_predictor = BasicMultStepUniVar(steps_past: int, steps_future: int)
loading_predictor.load_model(location: str)
loading_predictor.set_model_id(name: str)
```

2. Univariate-Multistep forecasting - Hybrid architectures

```python3
from imbrium.predictors.univarhybrid import HybridMultStepUniVar

predictor = HybridMultStepUniVar(sub_seq: int, steps_past: int, steps_future: int, data = pd.DataFrame(), scale: str = '')

# Choose between one of the architectures:

# predictor.create_cnnrnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',  layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})
# predictor.create_cnnlstm(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})
# predictor.create_cnngru(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})
# predictor.create_cnnbirnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})
# predictor.create_cnnbilstm(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})
# predictor.create_cnnbigru(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

# Fit the predictor object
predictor.fit_model(epochs: int, show_progress: int = 1, validation_split: float = 0.20, batch_size: int = 10)

# Have a look at the model performance - MSE based, more evaluation forms might be added on architecture level in the future
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

loading_predictor =  HybridMultStepUniVar(sub_seq: int, steps_past: int, steps_future: int)
loading_predictor.load_model(location: str)
loading_predictor.set_model_id(name: str)
```

### Multivariate Models:

1. Multivariate-Multistep forecasting - Standard architectures

```python3
from imbrium.predictors.multivarstandard import BasicMultStepMultVar

# please make sure that the target feature is the first variable in the feature list
predictor = BasicMultStepMultVar(steps_past: int, steps_future: int, data = pd.DataFrame(), features = [], scale: str = '')

# Choose between one of the architectures:

# predictor.create_mlp(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (25,'relu'), 'layer2': (25, 'relu')})
# predictor.create_rnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'), 'layer2': (50, 'relu')})
# predictor.create_lstm(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'), 'layer2': (50, 'relu')})
# predictor.create_gru(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'), 'layer2': (50, 'relu')})
# predictor.create_cnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu')})
# predictor.create_birnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})
# predictor.create_bilstm(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})
# predictor.create_bigru(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')})
# predictor.create_encdec_rnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})
# predictor.create_encdec_lstm(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})
# predictor.create_encdec_cnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (100, 'relu')})
# predictor.create_encdec_gru(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'), 'layer2': (50, 'relu'), 'layer3': (100, 'relu')})

# Fit the predictor object
predictor.fit_model(epochs: int, show_progress: int = 1, validation_split: float = 0.20, batch_size: int = 10)

# Have a look at the model performance - MSE based, more evaluation forms might be added on architecture level in the future
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

loading_predictor = BasicMultStepMultVar(steps_past: int, steps_future: int)
loading_predictor.load_model(location: str)
loading_predictor.set_model_id(name: str)
```
2. Multivariate-Multistep forecasting - Hybrid architectures

```python3
from imbrium.predictors.multivarhybrid import HybridMultStepMultVar

# please make sure that the target feature is the first variable in the feature list
predictor = HybridMultStepMultVar(sub_seq: int, steps_past: int, steps_future: int, data = pd.DataFrame(), features:list = [], scale: str = '')

# Choose between one of the architectures:

# predictor.create_cnnrnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})
# predictor.create_cnnlstm(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})
# predictor.create_cnngru(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})
# predictor.create_cnnbirnn(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})
# predictor.create_cnnbilstm(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',  layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})
# predictor.create_cnnbigru(optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error', layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')})

# Fit the predictor object
predictor.fit_model(epochs: int, show_progress: int = 1, validation_split: float = 0.20, batch_size: int = 10)

# Have a look at the model performance - MSE based, more evaluation forms might be added on architecture level in the future
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

loading_predictor =  HybridMultStepMultVar(sub_seq: int, steps_past: int, steps_future: int)
loading_predictor.load_model(location: str)
loading_predictor.set_model_id(name: str)
```
## References
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
