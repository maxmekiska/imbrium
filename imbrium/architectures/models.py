import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Bidirectional, TimeDistributed, GRU, SimpleRNN, RepeatVector


def mlp(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates MLP model by defining all layers with activation functions,
        optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(Dense(50, activation='relu', input_dim=input_shape))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def rnn(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates RNN model by defining all layers with activation functions,
        optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        SimpleRNN(
            40,
            activation='relu',
            return_sequences=True,
            input_shape=input_shape))
    model.add(SimpleRNN(50, activation='relu', return_sequences=True))
    model.add(SimpleRNN(50, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def lstm(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates LSTM model by defining all layers with activation functions,
        optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        LSTM(
            40,
            activation='relu',
            return_sequences=True,
            input_shape=input_shape))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def gru(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates GRU model by defining all layers with activation functions,
        optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(GRU(40,
                  activation='relu',
                  return_sequences=True,
                  input_shape=input_shape))
    model.add(GRU(50, activation='relu', return_sequences=True))
    model.add(GRU(50, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnn(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates the CNN model by defining all layers with activation functions,
        optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        Conv1D(
            filters=64,
            kernel_size=1,
            activation='relu',
            input_shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def birnn(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates a bidirectional RNN model by defining all layers with activation
        functions, optimizer, loss function and evaluation matrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        Bidirectional(
            SimpleRNN(
                50,
                activation='relu',
                return_sequences=True),
            input_shape=input_shape))
    model.add(SimpleRNN(50, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def bilstm(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates a bidirectional LSTM model by defining all layers with activation
        functions, optimizer, loss function and evaluation matrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        Bidirectional(
            LSTM(
                50,
                activation='relu',
                return_sequences=True),
            input_shape=input_shape))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def bigru(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates a bidirectional GRU model by defining all layers with activation
        functions, optimizer, loss function and evaluation matrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(Bidirectional(GRU(50, activation='relu',
              return_sequences=True), input_shape=input_shape))
    model.add(GRU(50, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def encdec_rnn(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int,
        repeat: int) -> object:
    '''Creates Encoder-Decoder RNN model by defining all layers with activation
        functions, optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        SimpleRNN(
            100,
            activation='relu',
            return_sequences=True,
            input_shape=input_shape))
    model.add(SimpleRNN(50, activation='relu'))
    model.add(RepeatVector(repeat))
    model.add(SimpleRNN(50, activation='relu', return_sequences=True))
    model.add(SimpleRNN(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(output_shape)))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def encdec_lstm(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int,
        repeat: int) -> object:
    '''Creates Encoder-Decoder LSTM model by defining all layers with activation
        functions, optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        LSTM(
            100,
            activation='relu',
            return_sequences=True,
            input_shape=input_shape))
    model.add(LSTM(50, activation='relu'))
    model.add(RepeatVector(repeat))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(output_shape)))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def encdec_cnn(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int,
        repeat: int) -> object:
    '''Creates Encoder-Decoder CNN model by defining all layers with activation
        functions, optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        Conv1D(
            filters=64,
            kernel_size=1,
            activation='relu',
            input_shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(repeat))
    model.add(GRU(50, activation='relu', return_sequences=True))
    model.add(GRU(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(output_shape)))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def encdec_gru(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int,
        repeat: int) -> object:
    '''Creates Encoder-Decoder GRU model by defining all layers with activation
        functions, optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(GRU(100,
                  activation='relu',
                  return_sequences=True,
                  input_shape=input_shape))
    model.add(GRU(50, activation='relu'))
    model.add(RepeatVector(repeat))
    model.add(GRU(50, activation='relu', return_sequences=True))
    model.add(GRU(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(output_shape)))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnnrnn(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates CNN-RNN hybrid model by defining all layers with activation
        functions, optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=64,
                kernel_size=1,
                activation='relu'),
            input_shape=input_shape))
    model.add(
        TimeDistributed(
            Conv1D(
                filters=32,
                kernel_size=1,
                activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(SimpleRNN(50, activation='relu', return_sequences=True))
    model.add(SimpleRNN(25, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnnlstm(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates CNN-LSTM hybrid model by defining all layers with activation
        functions, optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=64,
                kernel_size=1,
                activation='relu'),
            input_shape=input_shape))
    model.add(
        TimeDistributed(
            Conv1D(
                filters=32,
                kernel_size=1,
                activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(25, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnngru(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates CNN-GRU hybrid model by defining all layers with activation
        functions, optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=64,
                kernel_size=1,
                activation='relu'),
            input_shape=input_shape))
    model.add(
        TimeDistributed(
            Conv1D(
                filters=32,
                kernel_size=1,
                activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(GRU(50, activation='relu', return_sequences=True))
    model.add(GRU(25, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnnbirnn(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates CNN-BI-RNN hybrid model by defining all layers with activation
        functions, optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=64,
                kernel_size=1,
                activation='relu'),
            input_shape=input_shape))
    model.add(
        TimeDistributed(
            Conv1D(
                filters=32,
                kernel_size=1,
                activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(
        Bidirectional(
            SimpleRNN(
                50,
                activation='relu',
                return_sequences=True)))
    model.add(SimpleRNN(25, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnnbilstm(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates CNN-BI-LSTM hybrid model by defining all layers with activation
        functions, optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=64,
                kernel_size=1,
                activation='relu'),
            input_shape=input_shape))
    model.add(
        TimeDistributed(
            Conv1D(
                filters=32,
                kernel_size=1,
                activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(
        Bidirectional(
            LSTM(
                50,
                activation='relu',
                return_sequences=True)))
    model.add(LSTM(25, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def cnnbigru(
        optimizer: str,
        loss,
        metrics: str,
        input_shape: tuple,
        output_shape: int) -> object:
    '''Creates CNN-BI-GRU hybrid model by defining all layers with activation
        functions, optimizer, loss function and evaluation metrics.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            input_shape (tuple): Time series input shape.
            ouput_shape (int): Time series output shape.
        Returns:
            model (object): Returns compiled Keras model.
    '''
    model = keras.Sequential()
    model.add(
        TimeDistributed(
            Conv1D(
                filters=64,
                kernel_size=1,
                activation='relu'),
            input_shape=input_shape))
    model.add(
        TimeDistributed(
            Conv1D(
                filters=32,
                kernel_size=1,
                activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(GRU(50, activation='relu', return_sequences=True)))
    model.add(GRU(25, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
