import keras_core
from keras_core import regularizers
from keras_core.layers import (GRU, LSTM, Bidirectional, Conv1D, Dense,
                               Dropout, Flatten, MaxPooling1D, SimpleRNN,
                               TimeDistributed)


def mlp(
    optimizer: str,
    loss: str,
    metrics: str,
    dense_block_one: int,
    dense_block_two: int,
    dense_block_three: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(shape=(input_shape,)))
    for i in range(dense_block_one):
        model.add(
            Dense(
                layer_config[f"layer{layer_num}"]["config"]["neurons"],
                activation=layer_config[f"layer{layer_num}"]["config"]["activation"],
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"]["config"]["regularization"]
                ),
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for j in range(dense_block_two):
        model.add(
            Dense(
                layer_config[f"layer{layer_num}"]["config"]["neurons"],
                activation=layer_config[f"layer{layer_num}"]["config"]["activation"],
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"]["config"]["regularization"]
                ),
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for k in range(dense_block_three):
        if k == dense_block_three - 1:
            model.add(
                Dense(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
        else:
            model.add(
                Dense(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def rnn(
    optimizer: str,
    loss: str,
    metrics: str,
    rnn_block_one: int,
    rnn_block_two: int,
    rnn_block_three: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(rnn_block_one):
        model.add(
            SimpleRNN(
                layer_config[f"layer{layer_num}"]["config"]["neurons"],
                activation=layer_config[f"layer{layer_num}"]["config"]["activation"],
                return_sequences=True,
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"]["config"]["regularization"]
                ),
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for j in range(rnn_block_two):
        model.add(
            SimpleRNN(
                layer_config[f"layer{layer_num}"]["config"]["neurons"],
                activation=layer_config[f"layer{layer_num}"]["config"]["activation"],
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"]["config"]["regularization"]
                ),
                return_sequences=True,
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for k in range(rnn_block_three):
        if k == rnn_block_three - 1:
            model.add(
                SimpleRNN(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
        else:
            model.add(
                SimpleRNN(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                    return_sequences=True,
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def lstm(
    optimizer: str,
    loss: str,
    metrics: str,
    lstm_block_one: int,
    lstm_block_two: int,
    lstm_block_three: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(lstm_block_one):
        model.add(
            LSTM(
                layer_config[f"layer{layer_num}"]["config"]["neurons"],
                activation=layer_config[f"layer{layer_num}"]["config"]["activation"],
                return_sequences=True,
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"]["config"]["regularization"]
                ),
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for j in range(lstm_block_two):
        model.add(
            LSTM(
                layer_config[f"layer{layer_num}"]["config"]["neurons"],
                activation=layer_config[f"layer{layer_num}"]["config"]["activation"],
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"]["config"]["regularization"]
                ),
                return_sequences=True,
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for k in range(lstm_block_three):
        if k == lstm_block_three - 1:
            model.add(
                LSTM(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
        else:
            model.add(
                LSTM(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                    return_sequences=True,
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def cnn(
    optimizer: str,
    loss: str,
    metrics: str,
    conv_block_one: int,
    conv_block_two: int,
    dense_block_one: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(conv_block_one):
        model.add(
            Conv1D(
                filters=layer_config[f"layer{layer_num}"]["config"]["filters"],
                kernel_size=layer_config[f"layer{layer_num}"]["config"]["kernel_size"],
                activation=layer_config[f"layer{layer_num}"]["config"]["activation"],
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"]["config"]["regularization"]
                ),
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for j in range(conv_block_two):
        model.add(
            Conv1D(
                filters=layer_config[f"layer{layer_num}"]["config"]["filters"],
                kernel_size=layer_config[f"layer{layer_num}"]["config"]["kernel_size"],
                activation=layer_config[f"layer{layer_num}"]["config"]["activation"],
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"]["config"]["regularization"]
                ),
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    model.add(
        MaxPooling1D(pool_size=layer_config[f"layer{layer_num}"]["config"]["pool_size"])
    )
    model.add(Flatten())
    layer_num += 1
    for k in range(dense_block_one):
        if k == dense_block_one - 1:
            model.add(
                Dense(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
        else:
            model.add(
                Dense(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def gru(
    optimizer: str,
    loss: str,
    metrics: str,
    gru_block_one: int,
    gru_block_two: int,
    gru_block_three: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(gru_block_one):
        model.add(
            GRU(
                layer_config[f"layer{layer_num}"]["config"]["neurons"],
                activation=layer_config[f"layer{layer_num}"]["config"]["activation"],
                return_sequences=True,
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"]["config"]["regularization"]
                ),
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for j in range(gru_block_two):
        model.add(
            GRU(
                layer_config[f"layer{layer_num}"]["config"]["neurons"],
                activation=layer_config[f"layer{layer_num}"]["config"]["activation"],
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"]["config"]["regularization"]
                ),
                return_sequences=True,
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for k in range(gru_block_three):
        if k == gru_block_three - 1:
            model.add(
                GRU(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
        else:
            model.add(
                GRU(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                    return_sequences=True,
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def birnn(
    optimizer: str,
    loss: str,
    metrics: str,
    birnn_block_one: int,
    rnn_block_one: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(birnn_block_one):
        model.add(
            Bidirectional(
                SimpleRNN(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                ),
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for j in range(rnn_block_one):
        if j == rnn_block_one - 1:
            model.add(
                SimpleRNN(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
        else:
            model.add(
                SimpleRNN(
                    layer_config[f"layer{layer_num}"][0],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def bilstm(
    optimizer: str,
    loss: str,
    metrics: str,
    bilstm_block_one: int,
    lstm_block_one: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(bilstm_block_one):
        model.add(
            Bidirectional(
                LSTM(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    return_sequences=True,
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                ),
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for j in range(lstm_block_one):
        if j == lstm_block_one - 1:
            model.add(
                LSTM(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
        else:
            model.add(
                LSTM(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                    return_sequences=True,
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def bigru(
    optimizer: str,
    loss: str,
    metrics: str,
    bigru_block_one: int,
    gru_block_one: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(bigru_block_one):
        model.add(
            Bidirectional(
                GRU(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                    return_sequences=True,
                ),
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for j in range(gru_block_one):
        if j == gru_block_one - 1:
            model.add(
                GRU(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
        else:
            model.add(
                GRU(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                    return_sequences=True,
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def cnnrnn(
    optimizer: str,
    loss: str,
    metrics: str,
    conv_block_one: int,
    conv_block_two: int,
    rnn_block_one: int,
    rnn_block_two: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(conv_block_one):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config[f"layer{layer_num}"]["config"]["filters"],
                    kernel_size=layer_config[f"layer{layer_num}"]["config"][
                        "kernel_size"
                    ],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                ),
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for j in range(conv_block_two):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config[f"layer{layer_num}"]["config"]["filters"],
                    kernel_size=layer_config[f"layer{layer_num}"]["config"][
                        "kernel_size"
                    ],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    model.add(
        TimeDistributed(
            MaxPooling1D(
                pool_size=layer_config[f"layer{layer_num}"]["config"]["pool_size"]
            )
        )
    )
    model.add(TimeDistributed(Flatten()))
    layer_num += 1
    for k in range(rnn_block_one):
        model.add(
            SimpleRNN(
                layer_config[f"layer{layer_num}"]["config"]["neurons"],
                activation=layer_config[f"layer{layer_num}"]["config"]["activation"],
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"]["config"]["regularization"]
                ),
                return_sequences=True,
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
        layer_num += 1
    for l in range(rnn_block_two):
        if l == rnn_block_two - 1:
            model.add(
                SimpleRNN(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                )
            )
        else:
            model.add(
                SimpleRNN(
                    layer_config[f"layer{layer_num}"]["config"]["neurons"],
                    activation=layer_config[f"layer{layer_num}"]["config"][
                        "activation"
                    ],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"]["config"]["regularization"]
                    ),
                    return_sequences=True,
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"]["config"]["dropout"]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def cnnlstm(
    optimizer: str,
    loss: str,
    metrics: str,
    conv_block_one: int,
    conv_block_two: int,
    lstm_block_one: int,
    lstm_block_two: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(conv_block_one):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config[f"layer{layer_num}"][0],
                    kernel_size=layer_config[f"layer{layer_num}"][1],
                    activation=layer_config[f"layer{layer_num}"][2],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][3]
                    ),
                ),
                # input_shape=input_shape,
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][4]))
        layer_num += 1
    for j in range(conv_block_two):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config[f"layer{layer_num}"][0],
                    kernel_size=layer_config[f"layer{layer_num}"][1],
                    activation=layer_config[f"layer{layer_num}"][2],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][3]
                    ),
                )
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][4]))
        layer_num += 1
    model.add(
        TimeDistributed(MaxPooling1D(pool_size=layer_config[f"layer{layer_num}"]))
    )
    model.add(TimeDistributed(Flatten()))
    layer_num += 1
    for k in range(lstm_block_one):
        model.add(
            LSTM(
                layer_config[f"layer{layer_num}"][0],
                activation=layer_config[f"layer{layer_num}"][1],
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"][2]
                ),
                return_sequences=True,
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][3]))
        layer_num += 1
    for l in range(lstm_block_two):
        if l == lstm_block_two - 1:
            model.add(
                LSTM(
                    layer_config[f"layer{layer_num}"][0],
                    activation=layer_config[f"layer{layer_num}"][1],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][2]
                    ),
                )
            )
        else:
            model.add(
                LSTM(
                    layer_config[f"layer{layer_num}"][0],
                    activation=layer_config[f"layer{layer_num}"][1],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][2]
                    ),
                    return_sequences=True,
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"][3]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def cnngru(
    optimizer: str,
    loss: str,
    metrics: str,
    conv_block_one: int,
    conv_block_two: int,
    gru_block_one: int,
    gru_block_two: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(conv_block_one):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config[f"layer{layer_num}"][0],
                    kernel_size=layer_config[f"layer{layer_num}"][1],
                    activation=layer_config[f"layer{layer_num}"][2],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][3]
                    ),
                ),
                # input_shape=input_shape,
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][4]))
        layer_num += 1
    for j in range(conv_block_two):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config[f"layer{layer_num}"][0],
                    kernel_size=layer_config[f"layer{layer_num}"][1],
                    activation=layer_config[f"layer{layer_num}"][2],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][3]
                    ),
                )
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][4]))
        layer_num += 1
    model.add(
        TimeDistributed(MaxPooling1D(pool_size=layer_config[f"layer{layer_num}"]))
    )
    model.add(TimeDistributed(Flatten()))
    layer_num += 1
    for k in range(gru_block_one):
        model.add(
            GRU(
                layer_config[f"layer{layer_num}"][0],
                activation=layer_config[f"layer{layer_num}"][1],
                kernel_regularizer=regularizers.L2(
                    layer_config[f"layer{layer_num}"][2]
                ),
                return_sequences=True,
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][3]))
        layer_num += 1
    for l in range(gru_block_two):
        if l == gru_block_two - 1:
            model.add(
                GRU(
                    layer_config[f"layer{layer_num}"][0],
                    activation=layer_config[f"layer{layer_num}"][1],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][2]
                    ),
                )
            )
        else:
            model.add(
                GRU(
                    layer_config[f"layer{layer_num}"][0],
                    activation=layer_config[f"layer{layer_num}"][1],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][2]
                    ),
                    return_sequences=True,
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"][3]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def cnnbirnn(
    optimizer: str,
    loss: str,
    metrics: str,
    conv_block_one: int,
    conv_block_two: int,
    birnn_block_one: int,
    rnn_block_one: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(conv_block_one):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config[f"layer{layer_num}"][0],
                    kernel_size=layer_config[f"layer{layer_num}"][1],
                    activation=layer_config[f"layer{layer_num}"][2],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][3]
                    ),
                ),
                # input_shape=input_shape,
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][4]))
        layer_num += 1
    for j in range(conv_block_two):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config[f"layer{layer_num}"][0],
                    kernel_size=layer_config[f"layer{layer_num}"][1],
                    activation=layer_config[f"layer{layer_num}"][2],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][3]
                    ),
                )
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][4]))
        layer_num += 1
    model.add(
        TimeDistributed(MaxPooling1D(pool_size=layer_config[f"layer{layer_num}"]))
    )
    model.add(TimeDistributed(Flatten()))
    layer_num += 1
    for k in range(birnn_block_one):
        model.add(
            Bidirectional(
                SimpleRNN(
                    layer_config[f"layer{layer_num}"][0],
                    activation=layer_config[f"layer{layer_num}"][1],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][2]
                    ),
                    return_sequences=True,
                )
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][3]))
        layer_num += 1
    for l in range(rnn_block_one):
        if l == rnn_block_one - 1:
            model.add(
                SimpleRNN(
                    layer_config[f"layer{layer_num}"][0],
                    activation=layer_config[f"layer{layer_num}"][1],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][2]
                    ),
                )
            )
        else:
            model.add(
                SimpleRNN(
                    layer_config[f"layer{layer_num}"][0],
                    activation=layer_config[f"layer{layer_num}"][1],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][2]
                    ),
                    return_sequences=True,
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"][3]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def cnnbilstm(
    optimizer: str,
    loss: str,
    metrics: str,
    conv_block_one: int,
    conv_block_two: int,
    bilstm_block_one: int,
    lstm_block_one: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(conv_block_one):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config[f"layer{layer_num}"][0],
                    kernel_size=layer_config[f"layer{layer_num}"][1],
                    activation=layer_config[f"layer{layer_num}"][2],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][3]
                    ),
                ),
                # input_shape=input_shape,
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][4]))
        layer_num += 1
    for j in range(conv_block_two):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config[f"layer{layer_num}"][0],
                    kernel_size=layer_config[f"layer{layer_num}"][1],
                    activation=layer_config[f"layer{layer_num}"][2],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][3]
                    ),
                )
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][4]))
        layer_num += 1
    model.add(
        TimeDistributed(MaxPooling1D(pool_size=layer_config[f"layer{layer_num}"]))
    )
    model.add(TimeDistributed(Flatten()))
    layer_num += 1
    for j in range(bilstm_block_one):
        model.add(
            Bidirectional(
                LSTM(
                    layer_config[f"layer{layer_num}"][0],
                    activation=layer_config[f"layer{layer_num}"][1],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][2]
                    ),
                    return_sequences=True,
                )
            )
        )
        model.add(Dropout(layer_config[f"layer{layer_num}"][3]))
        layer_num += 1
    for k in range(lstm_block_one):
        if k == lstm_block_one - 1:
            model.add(
                LSTM(
                    layer_config[f"layer{layer_num}"][0],
                    activation=layer_config[f"layer{layer_num}"][1],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][2]
                    ),
                )
            )
        else:
            model.add(
                LSTM(
                    layer_config[f"layer{layer_num}"][0],
                    activation=layer_config[f"layer{layer_num}"][1],
                    kernel_regularizer=regularizers.L2(
                        layer_config[f"layer{layer_num}"][2]
                    ),
                    return_sequences=True,
                )
            )
            model.add(Dropout(layer_config[f"layer{layer_num}"][3]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model


def cnnbigru(
    optimizer: str,
    loss: str,
    metrics: str,
    conv_block_one: int,
    conv_block_two: int,
    bigru_block_one: int,
    gru_block_one: int,
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
    layer_num = 0
    model = keras_core.Sequential()
    model.add(keras_core.Input(input_shape))
    for i in range(conv_block_one):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config["layer0"][0],
                    kernel_size=layer_config["layer0"][1],
                    activation=layer_config["layer0"][2],
                    kernel_regularizer=regularizers.L2(layer_config["layer0"][3]),
                ),
                # input_shape=input_shape,
            )
        )
        model.add(Dropout(layer_config["layer0"][4]))
        layer_num += 1
    for j in range(conv_block_two):
        model.add(
            TimeDistributed(
                Conv1D(
                    filters=layer_config["layer1"][0],
                    kernel_size=layer_config["layer1"][1],
                    activation=layer_config["layer1"][2],
                    kernel_regularizer=regularizers.L2(layer_config["layer1"][3]),
                )
            )
        )
        model.add(Dropout(layer_config["layer1"][4]))
        layer_num += 1
    model.add(TimeDistributed(MaxPooling1D(pool_size=layer_config["layer2"])))
    model.add(TimeDistributed(Flatten()))
    layer_num += 1
    for k in range(bigru_block_one):
        model.add(
            Bidirectional(
                GRU(
                    layer_config["layer3"][0],
                    activation=layer_config["layer3"][1],
                    kernel_regularizer=regularizers.L2(layer_config["layer3"][2]),
                    return_sequences=True,
                )
            )
        )
        model.add(Dropout(layer_config["layer3"][3]))
        layer_num += 1
    for l in range(gru_block_one):
        if l == gru_block_one - 1:
            model.add(
                GRU(
                    layer_config["layer4"][0],
                    activation=layer_config["layer4"][1],
                    kernel_regularizer=regularizers.L2(layer_config["layer4"][2]),
                )
            )
        else:
            model.add(
                GRU(
                    layer_config["layer4"][0],
                    activation=layer_config["layer4"][1],
                    kernel_regularizer=regularizers.L2(layer_config["layer4"][2]),
                    return_sequences=True,
                )
            )
            model.add(Dropout(layer_config["layer4"][3]))
            layer_num += 1
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    return model
