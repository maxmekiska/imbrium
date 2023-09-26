import datetime
import os

from keras_core.callbacks import EarlyStopping, TensorBoard
from keras_core.saving import load_model
from numpy import array
from pandas import DataFrame

from imbrium.architectures.models import *
from imbrium.blueprints.abstract_multivariate import MultiVariateMultiStep
from imbrium.utils.optimizer import get_optimizer
from imbrium.utils.scaler import SCALER
from imbrium.utils.transformer import data_prep_multi, multistep_prep_standard


class BasePureMulti(MultiVariateMultiStep):
    """Implements deep neural networks based on multivariate multipstep
    standard predictors.

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
        self,
        steps_past: int,
        steps_future: int,
        data=DataFrame(),
        features=[],
        scale: str = "",
    ) -> object:
        """
        Parameters:
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            data (DataFrame): Input data for model training.
            features (list): List of features. First feature in list will be
            set to the target variable.
            scale (str): How to scale the data before making predictions.
        """
        self.scaler = SCALER[scale]
        self.model_id = ""
        self.optimizer = ""
        self.loss = ""
        self.metrics = ""

        if len(data) > 0:
            self.data = data_prep_multi(data, features, self.scaler)
            self.input_x, self.input_y = multistep_prep_standard(
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
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        dense_block_one: int = 1,
        dense_block_two: int = 1,
        dense_block_three: int = 1,
        layer_config: dict = {
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
    ):
        """Creates MLP model.
        Parameters:
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            dense_block_one (int): Number of layers in first dense block.
            dense_block_two (int): Number of layers in second dense block.
            dense_block_three (int): Number of layers in third dense block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("MLP")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        # necessary to account for hyperparameter optimization
        try:
            self.input_x = self.backup_input_x
        except:
            self.backup_input_x = self.input_x.copy()

        self.dimension = self.input_x.shape[1] * self.input_x.shape[2]

        self.input_x = self.input_x.reshape((self.input_x.shape[0], self.dimension))

        self.model = mlp(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            dense_block_one=dense_block_one,
            dense_block_two=dense_block_two,
            dense_block_three=dense_block_three,
            layer_config=layer_config,
            input_shape=self.input_x.shape[1],
            output_shape=self.input_y.shape[1],
        )

    def create_rnn(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        rnn_block_one: int = 1,
        rnn_block_two: int = 1,
        rnn_block_three: int = 1,
        layer_config: dict = {
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
    ):
        """Creates RNN model.
        Parameters:
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            rnn_block_one (int): Number of layers in first rnn block.
            rnn_block_two (int): Number of layers in second rnn block.
            rnn_block_three (int): Number of layers in third rnn block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("RNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = rnn(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            rnn_block_one=rnn_block_one,
            rnn_block_two=rnn_block_two,
            rnn_block_three=rnn_block_three,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], self.input_x.shape[2]),
            output_shape=self.input_y.shape[1],
        )

    def create_lstm(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        lstm_block_one: int = 1,
        lstm_block_two: int = 1,
        lstm_block_three: int = 1,
        layer_config: dict = {
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
    ):
        """Creates LSTM model.
        Parameters:
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            lstm_block_one (int): Number of layers in first lstm block.
            lstm_block_two (int): Number of layers in second lstm block.
            lstm_block_three (int): Number of layers in third lstm block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("LSTM")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = lstm(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            lstm_block_one=lstm_block_one,
            lstm_block_two=lstm_block_two,
            lstm_block_three=lstm_block_three,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], self.input_x.shape[2]),
            output_shape=self.input_y.shape[1],
        )

    def create_cnn(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        dense_block_one: int = 1,
        layer_config: dict = {
            "layer0": (
                64,
                1,
                "relu",
                0.0,
                0.0,
            ),  # (neurons, kernel_size, activation, regularization, dropout)
            "layer1": (
                32,
                1,
                "relu",
                0.0,
                0.0,
            ),  # (neurons, kernel_size, activation, regularization, dropout)
            "layer2": (2),  # pooling
            "layer3": (50, "relu", 0.0),  # (neurons, activation, regularization)
        },
    ):
        """Creates CNN model.
        Parameters:
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            conv_block_one (int): Number of layers in first conv block.
            conv_block_two (int): Number of layers in second conv block.
            dense_block_one (int): Number of layers in dense block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("CNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = cnn(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            dense_block_one=dense_block_one,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], self.input_x.shape[2]),
            output_shape=self.input_y.shape[1],
        )

    def create_gru(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        gru_block_one: int = 1,
        gru_block_two: int = 1,
        gru_block_three: int = 1,
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
    ):
        """Creates GRU model.
        Parameters:
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            gru_block_one (int): Number of layers in first gru block.
            gru_block_two (int): Number of layers in second gru block.
            gru_block_three (int): Number of layers in third gru block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("GRU")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = gru(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            gru_block_one=gru_block_one,
            gru_block_two=gru_block_two,
            gru_block_three=gru_block_three,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], self.input_x.shape[2]),
            output_shape=self.input_y.shape[1],
        )

    def create_birnn(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        birnn_block_one: int = 1,
        rnn_block_one: int = 1,
        layer_config: dict = {
            "layer0": (50, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0),
        },  # (neurons, activation, regularization, dropout)
    ):
        """Creates BI-RNN model.
        Parameters:
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            birnn_block_one (int): Number of layers in first birnn block.
            rnn_block_one (int): Number of layers in first rnn block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("BI-RNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = birnn(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            birnn_block_one=birnn_block_one,
            rnn_block_one=rnn_block_one,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], self.input_x.shape[2]),
            output_shape=self.input_y.shape[1],
        )

    def create_bilstm(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        bilstm_block_one: int = 1,
        lstm_block_one: int = 1,
        layer_config: dict = {
            "layer0": (50, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0),
        },
    ):
        """Creates BI-LSTM model.
        Parameters:
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            bilstm_block_one (int): Number of layers in first bilstm block.
            lstm_block_one (int): Number of layers in first lstm block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("BI-LSTM")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = bilstm(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            bilstm_block_one=bilstm_block_one,
            lstm_block_one=lstm_block_one,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], self.input_x.shape[2]),
            output_shape=self.input_y.shape[1],
        )

    def create_bigru(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        bigru_block_one: int = 1,
        gru_block_one: int = 1,
        layer_config: dict = {
            "layer0": (50, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0),
        },
    ):
        """Creates BI-GRU model.
        Parameters:
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            bigru_block_one (int): Number of layers in first bigru block.
            gru_block_one (int): Number of layers in first gru block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("BI-GRU")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = bigru(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            bigru_block_one=bigru_block_one,
            gru_block_one=gru_block_one,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], self.input_x.shape[2]),
            output_shape=self.input_y.shape[1],
        )

    def fit_model(
        self,
        epochs: int,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Trains the model on data provided. Perfroms validation.
        Parameters:
            epochs (int): Number of epochs to train the model.
            show_progress (int): Prints training progress.
            validation_split (float): Determines size of Validation data.
            board (bool): Create TensorBoard.
            callback_settings (dict): Create a Keras EarlyStopping object.
        """
        if callback_setting == {}:
            if board == True:
                callback_board = TensorBoard(
                    log_dir="logs/fit/"
                    + self.model_id
                    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                    histogram_freq=1,
                )
                self.details = self.model.fit(
                    self.input_x,
                    self.input_y,
                    validation_split=validation_split,
                    epochs=epochs,
                    verbose=show_progress,
                    callbacks=[callback_board],
                )
            else:
                callback_board = TensorBoard(
                    log_dir="logs/fit/"
                    + self.model_id
                    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                    histogram_freq=1,
                )
                self.details = self.model.fit(
                    self.input_x,
                    self.input_y,
                    validation_split=validation_split,
                    epochs=epochs,
                    verbose=show_progress,
                )

        else:
            if board == True:
                callback_board = TensorBoard(
                    log_dir="logs/fit/"
                    + self.model_id
                    + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                    histogram_freq=1,
                )
                callback = EarlyStopping(**callback_setting)
                self.details = self.model.fit(
                    self.input_x,
                    self.input_y,
                    validation_split=validation_split,
                    epochs=epochs,
                    verbose=show_progress,
                    callbacks=[callback, callback_board],
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
        """Returns performance details."""
        return self.details

    def predict(self, data: array) -> DataFrame:
        """Takes in a sequence of values and outputs a forecast.
        Parameters:
            data (array): Input sequence which needs to be forecasted.
        Returns:
            (DataFrame): Forecast for sequence provided.
        """
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

    def freeze(self, absolute_path: str = CURRENT_PATH):
        """Save the current model to the current directory.
        Parameters:
           absolute_path (str): Path to save model to.
        """
        self.model.save(absolute_path)

    def retrieve(self, location: str):
        """Load a keras model from the path specified.
        Parameters:
            location (str): Path of keras model location.
        """
        self.model = load_model(location)


class PureMulti(BasePureMulti):
    def create_fit_mlp(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        dense_block_one: int = 1,
        dense_block_two: int = 1,
        dense_block_three: int = 1,
        layer_config: dict = {
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
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates and trains a Multi-Layer-Perceptron model."""
        self.create_mlp(
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            dense_block_one=dense_block_one,
            dense_block_two=dense_block_two,
            dense_block_three=dense_block_three,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_rnn(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        rnn_block_one: int = 1,
        rnn_block_two: int = 1,
        rnn_block_three: int = 1,
        layer_config: dict = {
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
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates and trains a RNN model."""
        self.create_rnn(
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            rnn_block_one=rnn_block_one,
            rnn_block_two=rnn_block_two,
            rnn_block_three=rnn_block_three,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_lstm(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        lstm_block_one: int = 1,
        lstm_block_two: int = 1,
        lstm_block_three: int = 1,
        layer_config: dict = {
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
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates and trains a LSTM model."""
        self.create_lstm(
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            lstm_block_one=lstm_block_one,
            lstm_block_two=lstm_block_two,
            lstm_block_three=lstm_block_three,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_cnn(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        dense_block_one: int = 1,
        layer_config: dict = {
            "layer0": (
                64,
                1,
                "relu",
                0.0,
                0.0,
            ),  # (neurons, kernel_size, activation, regularization, dropout)
            "layer1": (
                32,
                1,
                "relu",
                0.0,
                0.0,
            ),  # (neurons, kernel_size, activation, regularization, dropout)
            "layer2": (2),  # pooling
            "layer3": (50, "relu", 0.0),  # (neurons, activation, regularization)
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates and trains a CNN model."""
        self.create_cnn(
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            dense_block_one=dense_block_one,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_gru(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        gru_block_one: int = 1,
        gru_block_two: int = 1,
        gru_block_three: int = 1,
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
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates and trains a GRU model."""
        self.create_gru(
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            gru_block_one=gru_block_one,
            gru_block_two=gru_block_two,
            gru_block_three=gru_block_three,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_birnn(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        birnn_block_one: int = 1,
        rnn_block_one: int = 1,
        layer_config: dict = {
            "layer0": (50, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0),
        },  # (neurons, activation, regularization, dropout)
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates and trains a BI-RNN model."""
        self.create_birnn(
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            birnn_block_one=birnn_block_one,
            rnn_block_one=rnn_block_one,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_bilstm(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        bilstm_block_one: int = 1,
        lstm_block_one: int = 1,
        layer_config: dict = {
            "layer0": (50, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates and trains a BI-LSTM model."""
        self.create_bilstm(
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            bilstm_block_one=bilstm_block_one,
            lstm_block_one=lstm_block_one,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_bigru(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        bigru_block_one: int = 1,
        gru_block_one: int = 1,
        layer_config: dict = {
            "layer0": (50, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates and trains a BI-GRU model."""
        self.create_bigru(
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            bigru_block_one=bigru_block_one,
            gru_block_one=gru_block_one,
            layer_config=layer_config,
        )
        self.fit_model(
            epochs=epochs,
            show_progress=show_progress,
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]


__all__ = ["PureMulti"]
