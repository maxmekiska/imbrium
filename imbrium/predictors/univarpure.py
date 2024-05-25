import datetime
import os
from typing import Tuple

from keras.callbacks import EarlyStopping, TensorBoard
from keras.saving import load_model
from numpy import array

from imbrium.architectures.models import (bigru, bilstm, birnn, cnn, gru, lstm,
                                          mlp, rnn)
from imbrium.blueprints.abstract_univariate import UniVariateMultiStep
from imbrium.utils.optimizer import get_optimizer
from imbrium.utils.transformer import (data_prep_uni,
                                       sequence_prep_standard_uni,
                                       train_test_split)


class BasePureUni(UniVariateMultiStep):
    """Implements neural network based univariate multipstep predictors."""

    CURRENT_PATH = os.getcwd()

    def __init__(
        self,
        target: array = array([]),
        evaluation_split: float = 0.10,  # train: 90%, test: 10%
        validation_split: float = 0.20,  # train: 72%, test: 10%, val: 18%
    ) -> object:
        """
        Parameters:
            target (array): Input target data numpy array.
            evaluation_split (float): train test split.
            validation_split (float): validation size of train set.
        """
        self.target = target
        self.evaluation_split = evaluation_split
        self.validation_split = validation_split
        self.model_id = ""
        self.optimizer = ""
        self.loss = ""
        self.metrics = ""

    def _model_intake_prep(self, steps_past: int, steps_future: int) -> None:
        """Private method that prepares feature and label data arrays for model intake."""
        if len(self.target) > 0:
            temp_data = data_prep_uni(self.target)
            self.input_x, self.input_y = sequence_prep_standard_uni(
                temp_data, steps_past, steps_future
            )
            self.input_x, self.input_x_test = train_test_split(
                self.input_x, test_size=self.evaluation_split
            )
            self.input_y, self.input_y_test = train_test_split(
                self.input_y, test_size=self.evaluation_split
            )
        else:
            pass

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
    def get_target(self) -> array:
        """Get original target data."""
        return self.target

    @property
    def get_target_shape(self) -> array:
        """Get shape of original target data."""
        return self.target.shape

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
        steps_past: int,
        steps_future: int,
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
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            dense_block_one (int): Number of dense layers in first block.
            dense_block_two (int): Number of dense layers in second block.
            dense_block_three (int): Number of dense layers in third block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("MLP")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(steps_past, steps_future)

        self.input_x = self.input_x.reshape(
            (self.input_x.shape[0], self.input_x.shape[1])
        )

        self.input_x_test = self.input_x_test.reshape(
            (self.input_x_test.shape[0], self.input_x_test.shape[1])
        )

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

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
        steps_past: int,
        steps_future: int,
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
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            rnn_block_one (int): Number of RNN layers in first block.
            rnn_block_two (int): Number of RNN layers in second block.
            rnn_block_three (int): Number of RNN layers in third block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("RNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = rnn(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            rnn_block_one=rnn_block_one,
            rnn_block_two=rnn_block_two,
            rnn_block_three=rnn_block_three,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_lstm(
        self,
        steps_past: int,
        steps_future: int,
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
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            lstm_block_one (int): Number of LSTM layers in first block.
            lstm_block_two (int): Number of LSTM layers in second block.
            lstm_block_three (int): Number of LSTM layers in third block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("LSTM")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = lstm(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            lstm_block_one=lstm_block_one,
            lstm_block_two=lstm_block_two,
            lstm_block_three=lstm_block_three,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_cnn(
        self,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        dense_block_one: int = 1,
        layer_config: dict = {
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
    ):
        """Creates CNN model.
        Parameters:
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            conv_block_one (int): Number of CNN layers in first block.
            conv_block_two (int): Number of CNN layers in second block.
            dense_block_one (int): Number of dense layers in first block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("CNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = cnn(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            dense_block_one=dense_block_one,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_gru(
        self,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        gru_block_one: int = 1,
        gru_block_two: int = 1,
        gru_block_three: int = 1,
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
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    ):
        """Creates GRU model.
        Parameters:
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            gru_block_one (int): Number of GRU layers in first block.
            gru_block_two (int): Number of GRU layers in second block.
            gru_block_three (int): Number of GRU layers in third block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("GRU")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = gru(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            gru_block_one=gru_block_one,
            gru_block_two=gru_block_two,
            gru_block_three=gru_block_three,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_birnn(
        self,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        birnn_block_one: int = 1,
        rnn_block_one: int = 1,
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
                }
            },
        },
    ):
        """Creates BI-RNN model.
        Parameters:
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            birnn_block_one (int): Number of BI-RNN layers in first block.
            rnn_block_one (int): Number of RNN layers in first block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("BI-RNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = birnn(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            birnn_block_one=birnn_block_one,
            rnn_block_one=rnn_block_one,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_bilstm(
        self,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        bilstm_block_one: int = 1,
        lstm_block_one: int = 1,
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
                }
            },
        },
    ):
        """Creates BI-LSTM model.
        Parameters:
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            bilstm_block_one (int): Number of BI-LSTM layers in first block.
            lstm_block_one (int): Number of LSTM layers in first block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("BI-LSTM")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = bilstm(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            bilstm_block_one=bilstm_block_one,
            lstm_block_one=lstm_block_one,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def create_bigru(
        self,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        bigru_block_one: int = 1,
        gru_block_one: int = 1,
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
                }
            },
        },
    ):
        """Creates BI-GRU model.
        Parameters:
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            bigru_block_one (int): Number of BI-GRU layers in first block.
            gru_block_one (int): Number of GRU layers in first block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("BI-GRU")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = bigru(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            bigru_block_one=bigru_block_one,
            gru_block_one=gru_block_one,
            layer_config=layer_config,
            input_shape=(self.input_x.shape[1], 1),
            output_shape=self.input_y.shape[1],
        )

    def fit_model(
        self,
        epochs: int,
        show_progress: int = 1,
        board: bool = False,
        batch_size=None,
        **callback_setting: dict,
    ):
        """Trains the model on data provided. Perfroms validation.
        Parameters:
            epochs (int): Number of epochs to train the model.
            show_progress (int): Prints training progress.
            board (bool): Creates TensorBoard.
            batch_size (float): Batch size.
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
                    validation_split=self.validation_split,
                    epochs=epochs,
                    verbose=show_progress,
                    callbacks=[callback_board],
                    shuffle=False,
                    batch_size=batch_size,
                )
            else:
                self.details = self.model.fit(
                    self.input_x,
                    self.input_y,
                    validation_split=self.validation_split,
                    epochs=epochs,
                    verbose=show_progress,
                    shuffle=False,
                    batch_size=batch_size,
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
                    validation_split=self.validation_split,
                    epochs=epochs,
                    verbose=show_progress,
                    callbacks=[callback, callback_board],
                    shuffle=False,
                    batch_size=batch_size,
                )
            else:
                callback = EarlyStopping(**callback_setting)
                self.details = self.model.fit(
                    self.input_x,
                    self.input_y,
                    validation_split=self.validation_split,
                    epochs=epochs,
                    verbose=show_progress,
                    callbacks=[callback],
                    shuffle=False,
                    batch_size=batch_size,
                )
        return self.details

    def evaluate_model(self):
        self.evaluation_details = self.model.evaluate(
            x=self.input_x_test, y=self.input_y_test
        )

        return self.evaluation_details

    def model_blueprint(self):
        """Prints a summary of the models layer structure."""
        self.model.summary()

    def show_performance(self):
        """Returns performance details."""
        return self.details

    def show_evaluation(self):
        """Returns performance details on test data."""
        return self.evaluation_details

    def predict(self, data: array) -> array:
        """Takes in a sequence of values and outputs a forecast.
        Parameters:
            data (array): Input sequence which needs to be forecasted.
        Returns:
            (array): Forecast for sequence provided.
        """
        data = data.reshape(-1, 1)

        dimension = data.shape[0] * data.shape[1]  # MLP case

        try:
            data = data.reshape((1, data.shape[0], data.shape[1]))  # All other models
            y_pred = self.model.predict(data, verbose=0)

        except BaseException:
            data = data.reshape((1, dimension))  # MLP case
            y_pred = self.model.predict(data, verbose=0)

        y_pred = y_pred.reshape(y_pred.shape[1], y_pred.shape[0])

        return y_pred

    def freeze(self, absolute_path: str = CURRENT_PATH):
        """Save the current model to the current directory.
        Parameters:
           absolute_path (str): Path to save model to.
        """
        self.model.save(absolute_path)

    def retrieve(self, location: str):
        """Load a keras model from the path specified.
        Parameters:
            location (str): Path of keras model location
        """
        self.model = load_model(location)


class PureUni(BasePureUni):
    def create_fit_mlp(
        self,
        steps_past: int,
        steps_future: int,
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
        board: bool = False,
        batch_size=None,
        **callback_setting: dict,
    ):
        """Creates and trains a Multi-Layer-Perceptron model."""
        self.create_mlp(
            steps_past=steps_past,
            steps_future=steps_future,
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
            board=board,
            batch_size=batch_size,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_rnn(
        self,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        rnn_block_one: int = 1,
        rnn_block_two: int = 1,
        rnn_block_three: int = 1,
        metrics: str = "mean_squared_error",
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
        board: bool = False,
        batch_size=None,
        **callback_setting: dict,
    ):
        """Creates and trains a Recurrent Neural Network model."""
        self.create_rnn(
            steps_past=steps_past,
            steps_future=steps_future,
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
            board=board,
            batch_size=batch_size,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_lstm(
        self,
        steps_past: int,
        steps_future: int,
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
        board: bool = False,
        batch_size=None,
        **callback_setting: dict,
    ):
        """Creates and trains a LSTM model."""
        self.create_lstm(
            steps_past=steps_past,
            steps_future=steps_future,
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
            board=board,
            batch_size=batch_size,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_cnn(
        self,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        dense_block_one: int = 1,
        layer_config: dict = {
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
        epochs: int = 100,
        show_progress: int = 1,
        board: bool = False,
        batch_size=None,
        **callback_setting: dict,
    ):
        """Creates and trains a Convolutional Neural Network model."""
        self.create_cnn(
            steps_past=steps_past,
            steps_future=steps_future,
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
            board=board,
            batch_size=batch_size,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_gru(
        self,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        gru_block_one: int = 1,
        gru_block_two: int = 1,
        gru_block_three: int = 1,
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
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
        epochs: int = 100,
        show_progress: int = 1,
        board: bool = False,
        batch_size=None,
        **callback_setting: dict,
    ):
        """Creates and trains a GRU model."""
        self.create_gru(
            steps_past=steps_past,
            steps_future=steps_future,
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
            board=board,
            batch_size=batch_size,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_birnn(
        self,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        birnn_block_one: int = 1,
        rnn_block_one: int = 1,
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
                }
            },
        },
        epochs: int = 100,
        show_progress: int = 1,
        board: bool = False,
        batch_size=None,
        **callback_setting: dict,
    ):
        """Creates and trains a Bidirectional RNN model."""
        self.create_birnn(
            steps_past=steps_past,
            steps_future=steps_future,
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
            board=board,
            batch_size=batch_size,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_bilstm(
        self,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        bilstm_block_one: int = 1,
        lstm_block_one: int = 1,
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
                }
            },
        },
        epochs: int = 100,
        show_progress: int = 1,
        board: bool = False,
        batch_size=None,
        **callback_setting: dict,
    ):
        """Creates and trains a Bidirectional LSTM model."""
        self.create_bilstm(
            steps_past=steps_past,
            steps_future=steps_future,
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
            board=board,
            batch_size=batch_size,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_bigru(
        self,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        bigru_block_one: int = 1,
        gru_block_one: int = 1,
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
                }
            },
        },
        epochs: int = 100,
        show_progress: int = 1,
        board: bool = False,
        batch_size=None,
        **callback_setting: dict,
    ):
        """Creates and trains a Bidirectional GRU model."""
        self.create_bigru(
            steps_past=steps_past,
            steps_future=steps_future,
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
            board=board,
            batch_size=batch_size,
            **callback_setting,
        )
        return self.details.history[metrics][-1]


__all__ = ["PureUni"]
