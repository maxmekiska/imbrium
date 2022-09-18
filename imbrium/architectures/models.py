import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Flatten,
    Conv1D,
    MaxPooling1D,
    Dropout,
    Bidirectional,
    TimeDistributed,
    GRU,
    SimpleRNN,
    RepeatVector,
)


def mlp(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates MLP model by defining all layers with activation functions,
    optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        Dense(
            layer_config["layer0"][0],
            activation=layer_config["layer0"][1],
            input_dim=input_shape,
        )
    )
    model.add(Dense(layer_config["layer1"][0], activation=layer_config["layer1"][1]))
    model.add(Dense(layer_config["layer2"][0], activation=layer_config["layer2"][1]))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def rnn(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates RNN model by defining all layers with activation functions,
    optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        SimpleRNN(
            layer_config["layer0"][0],
            activation=layer_config["layer0"][1],
            return_sequences=True,
            input_shape=input_shape,
        )
    )
    model.add(
        SimpleRNN(
            layer_config["layer1"][0],
            activation=layer_config["layer1"][1],
            return_sequences=True,
        )
    )
    model.add(
        SimpleRNN(layer_config["layer2"][0], activation=layer_config["layer2"][1])
    )
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def lstm(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates LSTM model by defining all layers with activation functions,
    optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        LSTM(
            layer_config["layer0"][0],
            activation=layer_config["layer0"][1],
            return_sequences=True,
            input_shape=input_shape,
        )
    )
    model.add(
        LSTM(
            layer_config["layer1"][0],
            activation=layer_config["layer1"][1],
            return_sequences=True,
        )
    )
    model.add(LSTM(layer_config["layer2"][0], activation=layer_config["layer2"][1]))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnn(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates the CNN model by defining all layers with activation functions,
    optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        Conv1D(
            filters=layer_config["layer0"][0],
            kernel_size=layer_config["layer0"][1],
            activation=layer_config["layer0"][2],
            input_shape=input_shape,
        )
    )
    model.add(
        Conv1D(
            filters=layer_config["layer1"][0],
            kernel_size=layer_config["layer1"][1],
            activation=layer_config["layer1"][2],
        )
    )
    model.add(MaxPooling1D(pool_size=layer_config["layer2"]))
    model.add(Flatten())
    model.add(Dense(layer_config["layer3"][0], activation=layer_config["layer3"][1]))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def gru(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates GRU model by defining all layers with activation functions,
    optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        GRU(
            layer_config["layer0"][0],
            activation=layer_config["layer0"][1],
            return_sequences=True,
            input_shape=input_shape,
        )
    )
    model.add(
        GRU(
            layer_config["layer1"][0],
            activation=layer_config["layer1"][1],
            return_sequences=True,
        )
    )
    model.add(GRU(layer_config["layer2"][0], activation=layer_config["layer2"][1]))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def birnn(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates a bidirectional RNN model by defining all layers with activation
    functions, optimizer, loss function and evaluation matrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        Bidirectional(
            SimpleRNN(
                layer_config["layer0"][0],
                activation=layer_config["layer0"][1],
                return_sequences=True,
            ),
            input_shape=input_shape,
        )
    )
    model.add(
        SimpleRNN(layer_config["layer1"][0], activation=layer_config["layer1"][1])
    )
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def bilstm(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates a bidirectional LSTM model by defining all layers with activation
    functions, optimizer, loss function and evaluation matrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        Bidirectional(
            LSTM(
                layer_config["layer0"][0],
                activation=layer_config["layer0"][1],
                return_sequences=True,
            ),
            input_shape=input_shape,
        )
    )
    model.add(LSTM(layer_config["layer1"][0], activation=layer_config["layer1"][1]))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def bigru(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates a bidirectional GRU model by defining all layers with activation
    functions, optimizer, loss function and evaluation matrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        Bidirectional(
            GRU(
                layer_config["layer0"][0],
                activation=layer_config["layer0"][1],
                return_sequences=True,
            ),
            input_shape=input_shape,
        )
    )
    model.add(GRU(layer_config["layer1"][0], activation=layer_config["layer1"][1]))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def encdec_rnn(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
    repeat: int,
) -> object:
    """Creates Encoder-Decoder RNN model by defining all layers with activation
    functions, optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        SimpleRNN(
            layer_config["layer0"][0],
            activation=layer_config["layer0"][1],
            return_sequences=True,
            input_shape=input_shape,
        )
    )
    model.add(
        SimpleRNN(layer_config["layer1"][0], activation=layer_config["layer1"][1])
    )
    model.add(RepeatVector(repeat))
    model.add(
        SimpleRNN(
            layer_config["layer2"][0],
            activation=layer_config["layer2"][1],
            return_sequences=True,
        )
    )
    model.add(
        SimpleRNN(
            layer_config["layer3"][0],
            activation=layer_config["layer3"][1],
            return_sequences=True,
        )
    )
    model.add(TimeDistributed(Dense(output_shape)))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def encdec_lstm(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
    repeat: int,
) -> object:
    """Creates Encoder-Decoder LSTM model by defining all layers with activation
    functions, optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        LSTM(
            layer_config["layer0"][0],
            activation=layer_config["layer0"][1],
            return_sequences=True,
            input_shape=input_shape,
        )
    )
    model.add(LSTM(layer_config["layer1"][0], activation=layer_config["layer1"][1]))
    model.add(RepeatVector(repeat))
    model.add(
        LSTM(
            layer_config["layer2"][0],
            activation=layer_config["layer2"][1],
            return_sequences=True,
        )
    )
    model.add(
        LSTM(
            layer_config["layer3"][0],
            activation=layer_config["layer3"][1],
            return_sequences=True,
        )
    )
    model.add(TimeDistributed(Dense(output_shape)))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def encdec_cnn(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
    repeat: int,
) -> object:
    """Creates Encoder-Decoder CNN model by defining all layers with activation
    functions, optimizer, loss function and evaluation metrics.
    Encoding via CNN and decoding via GRU.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        Conv1D(
            layer_config["layer0"][0],
            kernel_size=layer_config["layer0"][1],
            activation=layer_config["layer0"][2],
            input_shape=input_shape,
        )
    )
    model.add(
        Conv1D(
            filters=layer_config["layer1"][0],
            kernel_size=layer_config["layer1"][1],
            activation=layer_config["layer0"][2],
        )
    )
    model.add(MaxPooling1D(pool_size=layer_config["layer2"]))
    model.add(Flatten())
    model.add(RepeatVector(repeat))
    model.add(
        GRU(
            layer_config["layer3"][0],
            activation=layer_config["layer3"][1],
            return_sequences=True,
        )
    )
    model.add(
        GRU(
            layer_config["layer4"][0],
            activation=layer_config["layer4"][1],
            return_sequences=True,
        )
    )
    model.add(TimeDistributed(Dense(output_shape)))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def encdec_gru(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
    repeat: int,
) -> object:
    """Creates Encoder-Decoder GRU model by defining all layers with activation
    functions, optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        GRU(
            layer_config["layer0"][0],
            activation=layer_config["layer0"][1],
            return_sequences=True,
            input_shape=input_shape,
        )
    )
    model.add(GRU(layer_config["layer1"][0], activation=layer_config["layer1"][1]))
    model.add(RepeatVector(repeat))
    model.add(
        GRU(
            layer_config["layer2"][0],
            activation=layer_config["layer2"][1],
            return_sequences=True,
        )
    )
    model.add(
        GRU(
            layer_config["layer3"][0],
            activation=layer_config["layer3"][1],
            return_sequences=True,
        )
    )
    model.add(TimeDistributed(Dense(output_shape)))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnnrnn(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates CNN-RNN hybrid model by defining all layers with activation
    functions, optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer0"][0],
                kernel_size=layer_config["layer0"][1],
                activation=layer_config["layer0"][2],
            ),
            input_shape=input_shape,
        )
    )
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer1"][0],
                kernel_size=layer_config["layer1"][1],
                activation=layer_config["layer1"][2],
            )
        )
    )
    model.add(TimeDistributed(MaxPooling1D(pool_size=layer_config["layer2"])))
    model.add(TimeDistributed(Flatten()))
    model.add(
        SimpleRNN(
            layer_config["layer3"][0],
            activation=layer_config["layer3"][1],
            return_sequences=True,
        )
    )
    model.add(
        SimpleRNN(layer_config["layer4"][0], activation=layer_config["layer4"][1])
    )
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnnlstm(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates CNN-LSTM hybrid model by defining all layers with activation
    functions, optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer0"][0],
                kernel_size=layer_config["layer0"][1],
                activation=layer_config["layer0"][2],
            ),
            input_shape=input_shape,
        )
    )
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer1"][0],
                kernel_size=layer_config["layer1"][1],
                activation=layer_config["layer1"][2],
            )
        )
    )
    model.add(TimeDistributed(MaxPooling1D(pool_size=layer_config["layer2"])))
    model.add(TimeDistributed(Flatten()))
    model.add(
        LSTM(
            layer_config["layer3"][0],
            activation=layer_config["layer3"][1],
            return_sequences=True,
        )
    )
    model.add(LSTM(layer_config["layer4"][0], activation=layer_config["layer4"][1]))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnngru(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates CNN-GRU hybrid model by defining all layers with activation
    functions, optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer0"][0],
                kernel_size=layer_config["layer0"][1],
                activation=layer_config["layer0"][2],
            ),
            input_shape=input_shape,
        )
    )
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer1"][0],
                kernel_size=layer_config["layer1"][1],
                activation=layer_config["layer1"][2],
            )
        )
    )
    model.add(TimeDistributed(MaxPooling1D(pool_size=layer_config["layer2"])))
    model.add(TimeDistributed(Flatten()))
    model.add(
        GRU(
            layer_config["layer3"][0],
            activation=layer_config["layer3"][1],
            return_sequences=True,
        )
    )
    model.add(GRU(layer_config["layer4"][0], activation=layer_config["layer4"][1]))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnnbirnn(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates CNN-BI-RNN hybrid model by defining all layers with activation
    functions, optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer0"][0],
                kernel_size=layer_config["layer0"][1],
                activation=layer_config["layer0"][2],
            ),
            input_shape=input_shape,
        )
    )
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer1"][0],
                kernel_size=layer_config["layer1"][1],
                activation=layer_config["layer1"][2],
            )
        )
    )
    model.add(TimeDistributed(MaxPooling1D(pool_size=layer_config["layer2"])))
    model.add(TimeDistributed(Flatten()))
    model.add(
        Bidirectional(
            SimpleRNN(
                layer_config["layer3"][0],
                activation=layer_config["layer3"][1],
                return_sequences=True,
            )
        )
    )
    model.add(
        SimpleRNN(layer_config["layer4"][0], activation=layer_config["layer4"][1])
    )
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnnbilstm(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates CNN-BI-LSTM hybrid model by defining all layers with activation
    functions, optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer0"][0],
                kernel_size=layer_config["layer0"][1],
                activation=layer_config["layer0"][2],
            ),
            input_shape=input_shape,
        )
    )
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer1"][0],
                kernel_size=layer_config["layer1"][1],
                activation=layer_config["layer1"][2],
            )
        )
    )
    model.add(TimeDistributed(MaxPooling1D(pool_size=layer_config["layer2"])))
    model.add(TimeDistributed(Flatten()))
    model.add(
        Bidirectional(
            LSTM(
                layer_config["layer3"][0],
                activation=layer_config["layer3"][1],
                return_sequences=True,
            )
        )
    )
    model.add(LSTM(layer_config["layer4"][0], activation=layer_config["layer4"][1]))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnnbigru(
    optimizer: str,
    loss: str,
    metrics: str,
    layer_config: dict,
    input_shape: tuple,
    output_shape: int,
) -> object:
    """Creates CNN-BI-GRU hybrid model by defining all layers with activation
    functions, optimizer, loss function and evaluation metrics.
    Parameters:
        optimizer (str): Optimization algorithm.
        loss (str): Loss function.
        metrics (str): Performance measurement.
        layer_config (dict): Adjust neurons and activation functions.
        input_shape (tuple): Time series input shape.
        ouput_shape (int): Time series output shape.
    Returns:
        model (object): Returns compiled Keras model.
    """
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer0"][0],
                kernel_size=layer_config["layer0"][1],
                activation=layer_config["layer0"][2],
            ),
            input_shape=input_shape,
        )
    )
    model.add(
        TimeDistributed(
            Conv1D(
                filters=layer_config["layer1"][0],
                kernel_size=layer_config["layer1"][1],
                activation=layer_config["layer1"][2],
            )
        )
    )
    model.add(TimeDistributed(MaxPooling1D(pool_size=layer_config["layer2"])))
    model.add(TimeDistributed(Flatten()))
    model.add(
        Bidirectional(
            GRU(
                layer_config["layer3"][0],
                activation=layer_config["layer3"][1],
                return_sequences=True,
            )
        )
    )
    model.add(GRU(layer_config["layer4"][0], activation=layer_config["layer4"][1]))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
