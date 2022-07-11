from imbrium.blueprints.abstract_multivariate import MultiVariateMultiStep
from imbrium.architectures.models import *

import matplotlib.pyplot as plt
from numpy import array
from numpy import reshape
from numpy import empty
from numpy import dstack, vstack

from sklearn.preprocessing import (StandardScaler,
                                   MinMaxScaler,
                                   MaxAbsScaler,
                                   FunctionTransformer)

import pandas as pd
from pandas import DataFrame
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (LSTM,
                                     Dense,
                                     Flatten,
                                     Conv1D,
                                     MaxPooling1D,
                                     Dropout,
                                     Bidirectional,
                                     TimeDistributed,
                                     GRU,
                                     SimpleRNN)


class HybridMultStepMultVar(MultiVariateMultiStep):
    '''Implements neural network based multivariate multipstep hybrid predictors.

        Methods
        -------
        _scaling(self, method: str) -> object:
            Private method to return scikit scaling object.
        _data_prep(self, data: DataFrame, features: list) -> array:
            Private method to extract features and convert DataFrame to an array.
            IMPORTANT: First feature is set to the target variable.
        _sequence_prep(input_sequence: array, steps_past: int, steps_future: int)
        -> [(array, array, int)]:
            Private method to prepare data for predictor ingestion.
        _multistep_prep(self, input_sequence: array, sub_seq: int, steps_past: int,
        steps_future: int) -> [(array, array)]:
            Private method to account for multiple feature sequences.
        set_model_id(self, name: str):
            Setter method to change model id name.
        get_X_input(self) -> array:
            Getter method to retrieve transformed X input - training.
        get_X_input_shape(self) -> tuple:
            Getter method to retrieve transformed X shape.
        get_y_input(self) -> array:
            Getter method to retrieve transformed y input - target.
        get_y_input_shape(self) -> array:
            Getter method to retrieve transformed y input shape.
        get_loss(self) -> str:
            Getter method to retrieve used model loss.
        get_metrics(self) -> str:
            Getter method to retrieve model evaluation metrics used.
        create_cnnrnn(self, optimizer: str = 'adam',
        loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
        layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'),
        'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')}):
            Builds CNN-RNN structure.
        create_cnnlstm(self, optimizer: str = 'adam',
        loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
        layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'),
        'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')}):
            Builds CNN-LSTM structure.
        create_cnngru(self, optimizer: str = 'adam',
        loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
        layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'),
        'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')}):
            Builds CNN-GRU structure.
        create_cnnbirnn(self, optimizer: str = 'adam',
        loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
        layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'),
        'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')}):
            Buidls CNN bidirectional RNN structure.
        create_cnnbilstm(self, optimizer: str = 'adam',
        loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
        layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'),
        'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')}):
            Builds CNN bidirectional LSTM structure.
        create_cnnbigru(self, optimizer: str = 'adam',
        loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
        layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'),
        'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (25, 'relu')}):
            Builds CNN bidirectional GRU structure.
        fit_model(self, epochs: int, show_progress: int = 1):
            Fitting selected model to data provided.
        model_blueprint(self):
            Print blueprint of layer structure.
        show_performance(self):
            Evaluate and plot model performance.
        predict(self, data: array):
            Takes in input data and outputs model forecasts.
        save_model(self):
            Saves current ceras model to current directory.
        load_model(self, location: str):
            Load model from location specified.
    '''

    def __init__(
            self,
            sub_seq: int,
            steps_past: int,
            steps_future: int,
            data=pd.DataFrame(),
            features: list = [],
            scale: str = '') -> object:
        '''
            Parameters:
                sub_seq (int): Divide data into further subsequences.
                steps_past (int): Steps predictor will look backward.
                steps_future (int): Steps predictor will look forward.
                data (DataFrame): Input data for model training.
                features (list): List of features. First feature in list will be
                set to the target variable.
                scale (str): How to scale the data before making predictions.
        '''
        self.sub_seq = sub_seq
        self.steps_past = steps_past
        self.steps_future = steps_future
        self.loss = ''
        self.metrics = ''

        self.scaler = self._scaling(scale)

        self.model_id = ''
        self.sub_seq = sub_seq

        if len(data) > 0:
            self.data = self._data_prep(data, features)
            self.input_x, self.input_y, self.modified_back = self._multistep_prep(
                self.data, sub_seq, steps_past, steps_future)
        else:
            self.data = data

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
            scaler = FunctionTransformer(lambda x: (
                x - x.min()) / (x.max() - x.min()), validate=True)

        return scaler

    def _data_prep(self, data: DataFrame, features: list) -> array:
        ''' Private method to extract features and convert DataFrame to an array.
                Parameters:
                    data (DataFrame): DataFrame containing multi-feature data.
                    features (list): All features that should be considered.
                Returns:
                    data (array): Array containing sequences of selected features.
        '''
        data = data[features]

        target = array(data.iloc[:, 0])

        self.scaler.fit(data.iloc[:, 1:])
        scaled = self.scaler.transform(data.iloc[:, 1:])
        scaled = scaled.transpose()

        scaled = vstack((target, scaled))

        return scaled

    def _sequence_prep(self,
                       input_sequence: array,
                       sub_seq: int,
                       steps_past: int,
                       steps_future: int) -> [(array,
                                               array,
                                               int)]:
        '''Prepares data input into X and y sequences. Lenght of the X sequence
           is dertermined by steps_past while the length of y is determined by
           steps_future. In detail, the predictor looks at sequence X and
           predicts sequence y.
                Parameters:
                    input_sequence (array): Sequence that contains time series in
                    array format
                    sub_seq (int): Division into subsequences.
                    steps_past (int): Steps the predictor will look backward
                    steps_future (int): Steps the predictor will look forward
                Returns:
                    X (array): Array containing all looking back sequences
                    y (array): Array containing all looking forward sequences
        '''
        length = len(input_sequence)
        if length == 0:
            return (empty(shape=[steps_past, steps_past]), 0)
        X = []
        y = []
        if length <= steps_past:
            raise ValueError(
                'Input sequence is equal to or shorter than steps to look backwards')
        if steps_future <= 0:
            raise ValueError('Steps in the future need to be bigger than 0')

        for i in range(length):
            last = i + steps_past
            if last > length - steps_future:
                X.append(input_sequence[i:last])
                y.append(input_sequence[last - 1:last - 1 + steps_future])
                break
            X.append(input_sequence[i:last])
            # modification to use correct target y sequence
            y.append(input_sequence[last - 1:last - 1 + steps_future])
        y = array(y)
        X = array(X)
        modified_back = X.shape[1] // sub_seq
        X = X.reshape((X.shape[0], sub_seq, modified_back, 1))
        return X, y, modified_back  # special treatment to account for sub sequence division

    def _multistep_prep(self,
                        input_sequence: array,
                        sub_seq: int,
                        steps_past: int,
                        steps_future: int) -> [(array,
                                                array)]:
        '''This function prepares input sequences into a suitable input format
           for a multivariate multistep model. The first seqeunce in the array
           needs to be the target variable y.
                Parameters:
                    input_sequence (array): Sequence that contains time series
                    in array format
                    sub_seq (int): Devision into subsequences.
                    steps_past (int): Steps the predictor will look backward
                    steps_future (int): Steps the predictor will look forward
                Returns:
                    X (array): Array containing all looking back sequences
                    y (array): Array containing all looking forward sequences
        '''
        X = []
        Y = []
        for i in range(len(input_sequence)):
            if i == 0:  # target variable should be first sequence
                _, y, _ = self._sequence_prep(
                    input_sequence[0], sub_seq, steps_past, steps_future)
                Y.append(y)
                continue  # skip since target column not requiered in X array
            x, _, mod = self._sequence_prep(
                input_sequence[i], sub_seq, steps_past, steps_future)
            X.append(x)
        X = dstack(X)
        Y = Y[0]  # getting array out of list
        return X, Y, mod

    def set_model_id(self, name: str):
        '''Setter method to change model id field.
            Parameters:
                name (str): Name for selected Model.
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

    def create_cnnrnn(
        self,
        optimizer: str = 'adam',
        loss: str = 'mean_squared_error',
        metrics: str = 'mean_squared_error',
        layer_config={
            'layer0': (
            64,
            1,
            'relu'),
            'layer1': (
                32,
                1,
                'relu'),
            'layer2': (2),
            'layer3': (
                50,
                'relu'),
            'layer4': (
            25,
            'relu')}):
        '''Creates CNN-RNN hybrid model.
            Parameters:
                optimizer (str): Optimization algorithm.
                loss (str): Loss function.
                metrics (str): Performance measurement.
                layer_config (dict): Adjust neurons and acitivation functions.
        '''
        self.set_model_id('CNN-RNN')
        self.loss = loss
        self.metrics = metrics

        self.model = cnnrnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3]),
            output_shape=self.input_y.shape[1])

    def create_cnnlstm(
        self,
        optimizer: str = 'adam',
        loss: str = 'mean_squared_error',
        metrics: str = 'mean_squared_error',
        layer_config={
            'layer0': (
            64,
            1,
            'relu'),
            'layer1': (
                32,
                1,
                'relu'),
            'layer2': (2),
            'layer3': (
                50,
                'relu'),
            'layer4': (
            25,
            'relu')}):
        '''Creates CNN-LSTM hybrid model.
            Parameters:
                optimizer (str): Optimization algorithm.
                loss (str): Loss function.
                metrics (str): Performance measurement.
                layer_config (dict): Adjust neurons and acitivation functions.
        '''
        self.set_model_id('CNN-LSTM')
        self.loss = loss
        self.metrics = metrics

        self.model = cnnlstm(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3]),
            output_shape=self.input_y.shape[1])

    def create_cnngru(
        self,
        optimizer: str = 'adam',
        loss: str = 'mean_squared_error',
        metrics: str = 'mean_squared_error',
        layer_config={
            'layer0': (
            64,
            1,
            'relu'),
            'layer1': (
                32,
                1,
                'relu'),
            'layer2': (2),
            'layer3': (
                50,
                'relu'),
            'layer4': (
            25,
            'relu')}):
        '''Creates CNN-GRU hybrid model.
            Parameters:
                optimizer (str): Optimization algorithm.
                loss (str): Loss function.
                metrics (str): Performance measurement.
                layer_config (dict): Adjust neurons and acitivation functions.
        '''
        self.set_model_id('CNN-GRU')
        self.loss = loss
        self.metrics = metrics

        self.model = cnngru(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3]),
            output_shape=self.input_y.shape[1])

    def create_cnnbirnn(
        self,
        optimizer: str = 'adam',
        loss: str = 'mean_squared_error',
        metrics: str = 'mean_squared_error',
        layer_config={
            'layer0': (
            64,
            1,
            'relu'),
            'layer1': (
                32,
                1,
                'relu'),
            'layer2': (2),
            'layer3': (
                50,
                'relu'),
            'layer4': (
            25,
            'relu')}):
        '''Creates CNN-BI-RNN hybrid model.
            Parameters:
                optimizer (str): Optimization algorithm.
                loss (str): Loss function.
                metrics (str): Performance measurement.
                layer_config (dict): Adjust neurons and acitivation functions.
        '''
        self.set_model_id('CNN-BI-RNN')
        self.loss = loss
        self.metrics = metrics

        self.model = cnnbirnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3]),
            output_shape=self.input_y.shape[1])

    def create_cnnbilstm(
        self,
        optimizer: str = 'adam',
        loss: str = 'mean_squared_error',
        metrics: str = 'mean_squared_error',
        layer_config={
            'layer0': (
            64,
            1,
            'relu'),
            'layer1': (
                32,
                1,
                'relu'),
            'layer2': (2),
            'layer3': (
                50,
                'relu'),
            'layer4': (
            25,
            'relu')}):
        '''Creates CNN-BI-LSTM hybrid model.
            Parameters:
                optimizer (str): Optimization algorithm.
                loss (str): Loss function.
                metrics (str): Performance measurement.
                layer_config (dict): Adjust neurons and acitivation functions.
        '''
        self.set_model_id('CNN-BI-LSTM')
        self.loss = loss
        self.metrics = metrics

        self.model = cnnbilstm(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3]),
            output_shape=self.input_y.shape[1])

    def create_cnnbigru(
        self,
        optimizer: str = 'adam',
        loss: str = 'mean_squared_error',
        metrics: str = 'mean_squared_error',
        layer_config={
            'layer0': (
            64,
            1,
            'relu'),
            'layer1': (
                32,
                1,
                'relu'),
            'layer2': (2),
            'layer3': (
                50,
                'relu'),
            'layer4': (
            25,
            'relu')}):
        '''Creates CNN-BI-GRU hybrid model.
            Parameters:
                optimizer (str): Optimization algorithm.
                loss (str): Loss function.
                metrics (str): Performance measurement.
                layer_config (dict): Adjust neurons and acitivation functions.
        '''
        self.set_model_id('CNN-BI-GRU')
        self.loss = loss
        self.metrics = metrics

        self.model = cnnbigru(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3]),
            output_shape=self.input_y.shape[1])

    def fit_model(
            self,
            epochs: int,
            show_progress: int = 1,
            validation_split: float = 0.20,
            batch_size: int = 10):
        '''Trains the model on data provided. Perfroms validation.
            Parameters:
                epochs (int): Number of epochs to train the model.
                show_progress (int): Prints training progress.
                validation_split (float): Determines size of Validation data.
                batch_size (int): Batch size of input data.
        '''
        self.details = self.model.fit(
            self.input_x,
            self.input_y,
            validation_split=0.20,
            batch_size=10,
            epochs=epochs,
            verbose=show_progress)
        return self.details

    def model_blueprint(self):
        '''Prints a summary of the models layer structure.
        '''
        self.model.summary()

    def show_performance(self):
        '''Plots model loss, validation loss over time.
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
        self.scaler.fit(data)
        data = self.scaler.transform(data)

        shape_ = int((data.shape[1] * self.steps_past) / self.sub_seq)
        data = data.reshape(1, self.sub_seq, shape_, 1)

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
