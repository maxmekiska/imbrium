import os

import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import array, empty, reshape
from pandas import DataFrame
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (GRU, LSTM, Bidirectional, Conv1D, Dense,
                                     Dropout, Flatten, MaxPooling1D,
                                     RepeatVector, SimpleRNN, TimeDistributed)

from imbrium.architectures.models import *
from imbrium.blueprints.abstract_univariate import UniVariateMultiStep
from imbrium.utils.scaler import SCALER
from imbrium.utils.transformer import data_prep_uni, sequence_prep_standard_uni


class PureUni(UniVariateMultiStep):
    """Implements neural network based univariate multipstep predictors.

    Methods
    -------
    set_model_id(self, name: str):
        Setter method to change model id name.
    get_model_id(self) -> array:
        Getter method to retrieve model id used.
    get_X_input(self) -> array:
        Getter method to retrieve transformed X input - training.
    get_X_input_shape(self) -> tuple:
        Getter method to retrieve transformed X shape.
    get_y_input(self) -> array:
        Getter method to retrieve transformed y input - target.
    get_y_input_shape(self) -> array:
        Getter method to retrieve transformed y input shape.
    get_optimizer(self) -> str:
        Getter method to retrieve model optimizer used.
    get_loss(self) -> str:
        Getter method to retrieve used model loss.
    get_metrics(self) -> str:
        Getter method to retrieve model evaluation metrics used.
    create_mlp(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (25,'relu'),
    'layer2': (25, 'relu')}):
        Builds MLP structure.
    create_rnn(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'),
    'layer2': (50, 'relu')}):
        Builds RNN structure.
    create_lstm(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'),
    'layer2': (50, 'relu')}):
        Builds LSTM structure.
    create_cnn(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'),
    'layer2': (2), 'layer3': (50, 'relu')}):
        Builds CNN structure.
    create_gru(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (40, 'relu'), 'layer1': (50,'relu'),
    'layer2': (50, 'relu')}):
        Builds GRU structure.
    create_birnn(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')}):
        Builds bidirectional RNN structure.
    create_bilstm(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')}):
        Builds bidirectional LSTM structure.
    create_bigru(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')}):
        Builds bidirectional GRU structure.
    create_encdec_rnn(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'),
    'layer2': (50, 'relu'), 'layer3': (100, 'relu')}):
        Builds encoder decoder RNN structure.
    create_encdec_lstm(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'),
    'layer2': (50, 'relu'), 'layer3': (100, 'relu')}):
        Builds encoder decoder LSTM structure.
    create_encdec_gru(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (100, 'relu'), 'layer1': (50, 'relu'),
    'layer2': (50, 'relu'), 'layer3': (100, 'relu')}):
        Builds encoder decoder GRU structure.
    create_encdec_cnn(self, optimizer: str = 'adam',
    loss: str = 'mean_squared_error', metrics: str = 'mean_squared_error',
    layer_config: dict = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'),
    'layer2': (2), 'layer3': (50, 'relu'), 'layer4': (100, 'relu')}):
        Builds encoder decoder CNN structure.
    fit_model(self, epochs: int, show_progress: int = 1,
    validation_split: float = 0.20,
    **callback_setting: dict):
        Fitting model onto provided data.
    model_blueprint(self):
        Print blueprint of layer structure.
    show_performance(self):
        Evaluate and plot model performance.
    predict(self, data: array):
        Takes in input data and outputs model forecasts.
    save_model(self, absolute_path: str = CURRENT_PATH):
        Saves current Keras model to current directory.
    load_model(self, location: str):
        Load model from location specified.
    """

    CURRENT_PATH = os.getcwd()

    def __init__(
        self, steps_past: int, steps_future: int, data=DataFrame(), scale: str = ""
    ) -> object:
        """
        Parameters:
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            data (DataFrame): Input data for model training.
            scale (str): How to scale the data before making predictions.
        """
        self.scaler = SCALER[scale]
        self.model_id = ""
        self.optimizer = ""
        self.loss = ""
        self.metrics = ""

        if len(data) > 0:
            self.data = array(data)
            self.data = data_prep_uni(data, self.scaler)
            self.input_x, self.input_y = sequence_prep_standard_uni(
                self.data, steps_past, steps_future
            )
        else:
            self.data = data

    def set_model_id(self, name: str):
        """Setter method to change model id field.
        Parameters:
            name (str): Name for selected Model.
        """
        self.model_id = name

    @property
    def get_model_id(self) -> str:
        """Get model id."""
        return self.model_id

    @property
    def get_X_input(self) -> array:
        """Get transformed feature data."""
        return self.input_x

    @property
    def get_X_input_shape(self) -> tuple:
        """Get shape fo transformed feature data."""
        return self.input_x.shape

    @property
    def get_y_input(self) -> array:
        """Get transformed target data."""
        return self.input_y

    @property
    def get_y_input_shape(self) -> tuple:
        """Get shape fo transformed target data."""
        return self.input_y.shape

    @property
    def get_optimizer(self) -> str:
        """Get model optimizer."""
        return self.optimizer

    @property
    def get_loss(self) -> str:
        """Get loss function."""
        return self.loss

    @property
    def get_metrics(self) -> str:
        """Get metrics."""
        return self.metrics

    def create_mlp(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (
                50,
                "relu",
                0.0,
                0.0,
            ),  # (neurons, activation, regularization, dropout)
            "layer1": (
                25,
                "relu",
                0.0,
                0.0,
            ),  # (neurons, activation, regularization, dropout)
            "layer2": (25, "relu", 0.0),  # (neurons, activation, regularization)
        },
    ):
        """Creates MLP model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("MLP")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.input_x = self.input_x.reshape(
            (self.input_x.shape[0], self.input_x.shape[1])
        )

        self.model = mlp(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=self.input_x.shape[1],
            output_shape=self.input_y.shape[1],
        )

    def create_rnn(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (40, "relu"),
            "layer1": (50, "relu"),
            "layer2": (50, "relu"),
        },
    ):
        """Creates RNN model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("RNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = rnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_lstm(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (40, "relu"),
            "layer1": (50, "relu"),
            "layer2": (50, "relu"),
        },
    ):
        """Creates LSTM model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("LSTM")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = lstm(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_cnn(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (64, 1, "relu"),
            "layer1": (32, 1, "relu"),
            "layer2": (2),
            "layer3": (50, "relu"),
        },
    ):
        """Creates CNN model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("CNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = cnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_gru(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (40, "relu"),
            "layer1": (50, "relu"),
            "layer2": (50, "relu"),
        },
    ):
        """Creates GRU model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("GRU")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = gru(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_birnn(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {"layer0": (50, "relu"), "layer1": (50, "relu")},
    ):
        """Creates BI-RNN model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("BI-RNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = birnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_bilstm(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {"layer0": (50, "relu"), "layer1": (50, "relu")},
    ):
        """Creates BI-LSTM model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("BI-LSTM")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = bilstm(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_bigru(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {"layer0": (50, "relu"), "layer1": (50, "relu")},
    ):
        """Creates BI-GRU model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("BI-GRU")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = bigru(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_encdec_rnn(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (100, "relu"),
            "layer1": (50, "relu"),
            "layer2": (50, "relu"),
            "layer3": (100, "relu"),
        },
    ):
        """Creates Encoder-Decoder-RNN model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("Encoder-Decoder-RNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = encdec_rnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], self.input_x.shape[2]),
            output_shape=self.input_x.shape[2],
            repeat=self.input_y.shape[1],
        )

    def create_encdec_lstm(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (100, "relu"),
            "layer1": (50, "relu"),
            "layer2": (50, "relu"),
            "layer3": (100, "relu"),
        },
    ):
        """Creates Encoder-Decoder-LSTM model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("Encoder-Decoder-LSTM")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = encdec_lstm(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], self.input_x.shape[2]),
            output_shape=self.input_x.shape[2],
            repeat=self.input_y.shape[1],
        )

    def create_encdec_gru(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (100, "relu"),
            "layer1": (50, "relu"),
            "layer2": (50, "relu"),
            "layer3": (100, "relu"),
        },
    ):
        """Creates Encoder-Decoder-GRU model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("Encoder-Decoder-GRU")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = encdec_gru(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], self.input_x.shape[2]),
            output_shape=self.input_x.shape[2],
            repeat=self.input_y.shape[1],
        )

    def create_encdec_cnn(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (64, 1, "relu"),
            "layer1": (32, 1, "relu"),
            "layer2": (2),
            "layer3": (50, "relu"),
            "layer4": (100, "relu"),
        },
    ):
        """Creates Encoder-Decoder-CNN model.
        Encoding via CNN and Decoding via GRU.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("Encoder(CNN)-Decoder(GRU)")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.model = encdec_cnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], self.input_x.shape[2]),
            output_shape=self.input_x.shape[2],
            repeat=self.input_y.shape[1],
        )

    def fit_model(
        self,
        epochs: int,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Trains the model on data provided. Perfroms validation.
        Parameters:
            epochs (int): Number of epochs to train the model.
            show_progress (int): Prints training progress.
            validation_split (float): Determines size of Validation data.
            callback_settings (dict): Create a Keras EarlyStopping object.
        """
        if callback_setting == {}:
            self.details = self.model.fit(
                self.input_x,
                self.input_y,
                validation_split=validation_split,
                epochs=epochs,
                verbose=show_progress,
            )
        else:
            callback = EarlyStopping(**callback_setting)
            self.details = self.model.fit(
                self.input_x,
                self.input_y,
                validation_split=validation_split,
                epochs=epochs,
                verbose=show_progress,
                callbacks=[callback],
            )
        return self.details

    def model_blueprint(self):
        """Prints a summary of the models layer structure."""
        self.model.summary()

    def show_performance(self):
        """Plots model loss, validation loss over time."""
        information = self.details

        plt.plot(information.history["loss"], color="black")
        plt.plot(information.history["val_loss"], color="teal")
        plt.title(self.model_id + " Model Loss")
        plt.ylabel(self.loss)
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper right")
        plt.tight_layout()
        plt.show()

    def predict(self, data: array) -> DataFrame:
        """Takes in a sequence of values and outputs a forecast.
        Parameters:
            data (array): Input sequence which needs to be forecasted.
        Returns:
            (DataFrame): Forecast for sequence provided.
        """
        data = array(data)
        data = data.reshape(-1, 1)

        self.scaler.fit(data)
        data = self.scaler.transform(data)

        dimension = data.shape[0] * data.shape[1]  # MLP case

        try:
            data = data.reshape((1, data.shape[0], data.shape[1]))  # All other models
            y_pred = self.model.predict(data, verbose=0)

        except BaseException:
            data = data.reshape((1, dimension))  # MLP case
            y_pred = self.model.predict(data, verbose=0)

        y_pred = y_pred.reshape(y_pred.shape[1], y_pred.shape[0])

        return DataFrame(y_pred, columns=[f"{self.model_id}"])

    def save_model(self, absolute_path: str = CURRENT_PATH):
        """Save the current model to the current directory.
        Parameters:
           absolute_path (str): Path to save model to.
        """
        self.model.save(absolute_path)

    def load_model(self, location: str):
        """Load a keras model from the path specified.
        Parameters:
            location (str): Path of keras model location
        """
        self.model = keras.models.load_model(location)


class OptimizePureUni(PureUni):
    def create_fit_mlp(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (
                50,
                "relu",
                0.0,
                0.0,
            ),  # (neurons, activation, regularization, dropout)
            "layer1": (
                25,
                "relu",
                0.0,
                0.0,
            ),  # (neurons, activation, regularization, dropout)
            "layer2": (25, "relu", 0.0),  # (neurons, activation, regularization)
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a Multi-Layer-Perceptron model."""
        self.create_mlp(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_rnn(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (
                40,
                "relu",
                0.0,
                0.0,
            ),  # (neurons, activation, regularization, dropout)
            "layer1": (
                50,
                "relu",
                0.0,
                0.0,
            ),  # (neurons, activation, regularization, dropout)
            "layer2": (50, "relu", 0.0),  # (neurons, activation, regularization)
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a Recurrent Neural Network model."""
        self.create_rnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_lstm(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (40, "relu"),
            "layer1": (50, "relu"),
            "layer2": (50, "relu"),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a LSTM model."""
        self.create_lstm(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_cnn(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (64, 1, "relu"),
            "layer1": (32, 1, "relu"),
            "layer2": (2),
            "layer3": (50, "relu"),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a Convolutional Neural Network model."""
        self.create_cnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_gru(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (40, "relu"),
            "layer1": (50, "relu"),
            "layer2": (50, "relu"),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a GRU model."""
        self.create_gru(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_birnn(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {"layer0": (50, "relu"), "layer1": (50, "relu")},
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a Bidirectional RNN model."""
        self.create_birnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_bilstm(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {"layer0": (50, "relu"), "layer1": (50, "relu")},
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a Bidirectional LSTM model."""
        self.create_bilstm(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_bigru(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {"layer0": (50, "relu"), "layer1": (50, "relu")},
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a Bidirectional GRU model."""
        self.create_bigru(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_encdec_rnn(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (100, "relu"),
            "layer1": (50, "relu"),
            "layer2": (50, "relu"),
            "layer3": (100, "relu"),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a Encoder-Decoder RNN model."""
        self.create_encdec_rnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_encdec_lstm(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (100, "relu"),
            "layer1": (50, "relu"),
            "layer2": (50, "relu"),
            "layer3": (100, "relu"),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a Encoder-Decoder LSTM model."""
        self.create_encdec_lstm(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_encdec_gru(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (100, "relu"),
            "layer1": (50, "relu"),
            "layer2": (50, "relu"),
            "layer3": (100, "relu"),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a Encoder-Decoder GRU model."""
        self.create_encdec_gru(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_encdec_cnn(
        self,
        optimizer: str = "adam",
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        layer_config: dict = {
            "layer0": (64, 1, "relu"),
            "layer1": (32, 1, "relu"),
            "layer2": (2),
            "layer3": (50, "relu"),
            "layer4": (100, "relu"),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict,
    ):
        """Creates and trains a Encoder-Decoder CNN model."""
        self.create_encdec_cnn(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            **callback_setting,
        )
        return self.details.history[metrics][-1]
