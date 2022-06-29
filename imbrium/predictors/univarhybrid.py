from imbrium.predictors.blueprints_predictors.abstract_univariate import UniVariateMultiStep

import matplotlib.pyplot as plt
from numpy import array
from numpy import reshape
import pandas as pd
from pandas import DataFrame
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, FunctionTransformer

import pandas as pd
from pandas import DataFrame
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Bidirectional, TimeDistributed, GRU, SimpleRNN

class HybridMultStepUniVar(UniVariateMultiStep):
    '''Implements neural network based univariate multipstep hybrid predictors.

        Methods
        -------
        _scaling(self, method: str) -> object:
            Private method to scale input data.
        _sequence_prep(input_sequence: array, steps_past: int, steps_future: int, scale: str = 'standard') -> [(array, array, int)]:
            Private method to prepare data for predictor ingestion.
        set_model_id(self, name: str):
            Setter method to change model id name.
        create_cnnlstm(self):
            Builds CNN-LSTM structure.
        fit_model(self, epochs: int, show_progress: int = 1):
            Training the in the prior defined model. Count of epochs need to be defined.
        model_blueprint(self):
            Print blueprint of layer structure.
        show_performance(self):
            Evaluate and plot model performance.
        predict(self, data: array, scale: str = 'standard'):
            Takes in input data and outputs model forecasts.
        save_model(self):
            Saves current ceras model to current directory.
        load_model(self, location: str):
            Load model from location specified.
    '''
    def __init__(self, sub_seq: int, steps_past: int, steps_future: int, data = pd.DataFrame(), scale: str = '') -> object:
        '''
            Parameters:
                sub_seq (int): Further division of given steps a predictor will look backward.
                steps_past (int): Steps predictor will look backward.
                steps_future (int): Steps predictor will look forward.
                data (array): Input data for model training. Default is empty to enable loading pre-trained models.
        '''
        self.sub_seq = sub_seq
        self.steps_past = steps_past
        self.steps_future = steps_future
        self.loss = ''
        self.metrics = ''

        self.scaler = self._scaling(scale)


        if len(data) > 0:
            self.data = array(data)
            self.data = self._data_prep(data)
            self.input_x, self.input_y, self.modified_back = self._sequence_prep(self.data, sub_seq, steps_past, steps_future)
        else:
            self.data = data

        self.model_id = '' # to identify model (example: name)

    def _scaling(self, method: str) -> object:
        '''Scales data accordingly.
            Parameters:
                method (str): Scaling method.
            Returns:
                scaler (object): Returns scikit learn scaler object.
        '''
        if method == '':
            scaler = FunctionTransformer(lambda x: x, validate=True)
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'maxabs':
            scaler = MaxAbsScaler()
        elif method == 'normalize':
            scaler = FunctionTransformer(lambda x: (x - x.min()) / (x.max() - x.min()), validate= True)

        return scaler

    def _data_prep(self, data: DataFrame) -> array:
        '''Prepares data input for model intake. Applies scaling to data.
            Parameters:
                data (DataFrame): Input time series.
            Returns:
                scaled (array): Scaled input time series.
        '''
        data = array(data).reshape(-1, 1)

        self.scaler.fit(data)
        scaled = self.scaler.transform(data)

        return scaled

    def _sequence_prep(self, input_sequence: array, sub_seq: int, steps_past: int, steps_future: int) -> [(array, array, int)]:
        '''Prepares data input into X and y sequences. Length of the X sequence is determined by steps_past while the length of y is determined by steps_future. In detail, the predictor looks at sequence X and predicts sequence y.
            Parameters:
                input_sequence (array): Sequence that contains time series in array format
                sub_seq (int): Further division of given steps a predictor will look backward.
                steps_past (int): Steps the predictor will look backward
                steps_future (int): Steps the predictor will look forward
            Returns:
                X (array): Array containing all looking back sequences
                y (array): Array containing all looking forward sequences
                modified_back (int): Modified looking back sequence length
        '''
        length = len(input_sequence)
        if length == 0:
            return (0, 0, steps_past // sub_seq)
        X = []
        y = []
        if length <= steps_past:
            raise ValueError('Input sequence is equal to or shorter than steps to look backwards')
        if steps_future <= 0:
            raise ValueError('Steps in the future need to be bigger than 0')

        for i in range(length):
            last = i + steps_past
            if last > length - steps_future:
                break
            X.append(input_sequence[i:last])
            y.append(input_sequence[last:last + steps_future])
        y = array(y)
        X = array(X)
        modified_back = X.shape[1]//sub_seq
        X = X.reshape((X.shape[0], sub_seq, modified_back, 1))
        return X, y, modified_back # special treatment to account for sub sequence division

    def set_model_id(self, name: str):
        '''Setter method to change model id field.
        '''
        self.model_id = name

    @property
    def get_X_input(self) -> array:
        '''Get transformed feature data.
        '''
        return self.input_x

    @property
    def get_X_input_shape(self) -> tuple:
        '''Get shape fo transformed feature data.
        '''
        return self.input_x.shape

    @property
    def get_y_input(self) -> array:
        '''Get transformed target data.
        '''
        return self.input_y

    @property
    def get_y_input_shape(self) -> tuple:
        '''Get shape fo transformed target data.
        '''
        return self.input_y.shape

    @property
    def get_loss(self) -> str:
        '''Get loss function.
        '''
        return self.loss

    @property
    def get_metrics(self) -> str:
        '''Get metrics.
        '''
        return self.metrics

    def create_cnnrnn(self, optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error'):
        '''Creates CNN-RNN hybrid model by defining all layers with activation functions, optimizer, loss function and evaluation metrics.
        '''
        self.set_model_id('CNN-RNN')
        self.loss = loss
        self.metrics = metrics

        self.model = keras.Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,self.modified_back, 1)))
        self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=1, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(SimpleRNN(50, activation='relu', return_sequences=True))
        self.model.add(SimpleRNN(25, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def create_cnnlstm(self, optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error'):
        '''Creates CNN-LSTM hybrid model by defining all layers with activation functions, optimizer, loss function and evaluation metrics.
        '''
        self.set_model_id('CNN-LSTM')
        self.loss = loss
        self.metrics = metrics

        self.model = keras.Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,self.modified_back, 1)))
        self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=1, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(50, activation='relu', return_sequences=True))
        self.model.add(LSTM(25, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def create_cnngru(self, optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error'):
        '''Creates CNN-GRU hybrid model by defining all layers with activation functions, optimizer, loss function and evaluation metrics.
        '''
        self.set_model_id('CNN-GRU')
        self.loss = loss
        self.metrics = metrics

        self.model = keras.Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,self.modified_back, 1)))
        self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=1, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(GRU(50, activation='relu', return_sequences=True))
        self.model.add(GRU(25, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def create_cnnbirnn(self, optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error'):
        '''Creates CNN-Bidirectional-RNN hybrid model by defining all layers with activation functions, optimizer, loss function and evaluation metrics.
        '''
        self.set_model_id('CNN-Bi-RNN')
        self.loss = loss
        self.metrics = metrics

        self.model = keras.Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,self.modified_back, 1)))
        self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=1, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(Bidirectional(SimpleRNN(50, activation='relu', return_sequences=True)))
        self.model.add(SimpleRNN(25, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def create_cnnbilstm(self, optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error'):
        '''Creates CNN-Bidirectional-LSTM hybrid model by defining all layers with activation functions, optimizer, loss function and evaluation metrics.
        '''
        self.set_model_id('CNN-Bi-LSTM')
        self.loss = loss
        self.metrics = metrics

        self.model = keras.Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,self.modified_back, 1)))
        self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=1, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True)))
        self.model.add(LSTM(25, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def create_cnnbigru(self, optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error'):
        '''Creates CNN-Bidirectional-GRU hybrid model by defining all layers with activation functions, optimizer, loss function and evaluation metrics.
        '''
        self.set_model_id('CNN-Bi-GRU')
        self.loss = loss
        self.metrics = metrics

        self.model = keras.Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None,self.modified_back, 1)))
        self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=1, activation='relu')))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(Bidirectional(GRU(50, activation='relu', return_sequences=True)))
        self.model.add(GRU(25, activation='relu'))
        self.model.add(Dense(self.input_y.shape[1]))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit_model(self, epochs: int, show_progress: int = 1, validation_split=0.20, batch_size = 10):
        '''Trains the model on data provided. Perfroms validation.
            Parameters:
                epochs (int): Number of epochs to train the model.
                show_progress (int): Prints training progress.
        '''
        self.details = self.model.fit(self.input_x, self.input_y, validation_split=validation_split, batch_size = batch_size, epochs = epochs, verbose=show_progress)
        return self.details

    def model_blueprint(self):
        '''Prints a summary of the models layer structure.
        '''
        self.model.summary()

    def show_performance(self):
        '''Plots:
        1. Models mean squared error of trainings and validation data. (Model loss)
        '''
        information = self.details

        plt.plot(information.history['loss'])
        plt.plot(information.history['val_loss'])
        plt.title(self.model_id + ' Model Loss')
        plt.ylabel(self.loss)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.tight_layout()
        plt.show()

    def predict(self, data: array) -> DataFrame:
        '''Takes in a sequence of values and outputs a forecast.
            Parameters:
                data (array): Input sequence which needs to be forecasted.
            Returns:
                (DataFrame): Forecast for sequence provided.
        '''
        data = array(data)
        data = data.reshape(-1, 1)

        self.scaler.fit(data)
        data = self.scaler.transform(data)

        shape_ = int((data.shape[1] * self.steps_past) / self.sub_seq)
        data = data.reshape(1, self.sub_seq, shape_ , 1)

        y_pred = self.model.predict(data, verbose=0)

        y_pred = y_pred.reshape(y_pred.shape[1], y_pred.shape[0])

        return pd.DataFrame(y_pred, columns=[f'{self.model_id}'])

    def save_model(self):
        '''Save the current model to the current directory.
        '''
        self.model.save(os.path.abspath(os.getcwd()))

    def load_model(self, location: str):
        '''Load a keras model from the path specified.
            Parameters:
                location (str): Path of keras model location
        '''
        self.model = keras.models.load_model(location)
