import datetime
import os
from typing import Any

import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.saving import load_model

from imbrium.architectures.models import (cnnbigru, cnnbilstm, cnnbirnn,
                                          cnngru, cnnlstm, cnnrnn)
from imbrium.blueprints.abstract_multivariate import MultiVariateMultiStep
from imbrium.utils.optimizer import get_optimizer
from imbrium.utils.transformer import (data_prep_multi, multistep_prep_hybrid,
                                       train_test_split)


class BaseHybridMulti(MultiVariateMultiStep):
    """Implements neural network based multivariate multipstep hybrid predictors."""

    CURRENT_PATH = os.getcwd()

    def __init__(
        self,
        target: np.ndarray = np.array([]),
        features: np.ndarray = np.array([]),
        evaluation_split: float = 0.10,  # train: 90%, test: 10%
        validation_split: float = 0.20,  # train: 72%, test: 10%, val: 18%
    ) -> object:
        """
        Parameters:
            target (array): Input target array.
            features (array): Input feature array.
            evaluation_split (float): train test split.
            validation_split (float): validation size of train set.
        """
        self.target = target
        self.features = features
        self.evaluation_split = evaluation_split
        self.validation_split = validation_split
        self.optimizer = ""
        self.loss = ""
        self.metrics = ""

        self.model_id = ""

    def _model_intake_prep(
        self, sub_seq: int, steps_past: int, steps_future: int
    ) -> None:
        """Private method that prepares feature and label data arrays for model intake."""
        self.steps_past = steps_past
        self.sub_seq = sub_seq
        if len(self.target) > 0:
            temp_data = data_prep_multi(self.target, self.features)
            self.input_x, self.input_y, self.modified_back = multistep_prep_hybrid(
                temp_data, sub_seq, steps_past, steps_future
            )
            self.input_x, self.input_x_test = train_test_split(
                self.input_x, test_size=self.evaluation_split
            )
            self.input_y, self.input_y_test = train_test_split(
                self.input_y, test_size=self.evaluation_split
            )
        else:
            pass

    def set_model_id(self, name: str) -> None:
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
    def get_target(self) -> np.ndarray:
        """Get original target data."""
        return self.target

    @property
    def get_target_shape(self) -> np.ndarray:
        """Get shape of original target data."""
        return self.target.shape

    @property
    def get_X_input(self) -> np.ndarray:
        """Get transformed feature data."""
        return self.input_x

    @property
    def get_X_input_shape(self) -> tuple[int, int]:
        """Get shape fo transformed feature data."""
        return self.input_x.shape

    @property
    def get_y_input(self) -> np.ndarray:
        """Get transformed target data."""
        return self.input_y

    @property
    def get_y_input_shape(self) -> tuple[int, int]:
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

    def create_cnnrnn(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        rnn_block_one: int = 1,
        rnn_block_two: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    ) -> None:
        """Creates CNN-RNN hybrid model.
        Parameters:
            sub_seq (int): Divide data into further subsequences.
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            conv_block_one (int): Number of convolutional layers in first block.
            conv_block_two (int): Number of convolutional layers in second block.
            rnn_block_one (int): Number of RNN layers in first block.
            rnn_block_two (int): Number of RNN layers in second block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("CNN-RNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(sub_seq, steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = cnnrnn(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            rnn_block_one=rnn_block_one,
            rnn_block_two=rnn_block_two,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3],
            ),
            output_shape=self.input_y.shape[1],
        )

    def create_cnnlstm(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        lstm_block_one: int = 1,
        lstm_block_two: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    ) -> None:
        """Creates CNN-LSTM hybrid model.
        Parameters:
            sub_seq (int): Divide data into further subsequences.
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            conv_block_one (int): Number of convolutional layers in first block.
            conv_block_two (int): Number of convolutional layers in second block.
            lstm_block_one (int): Number of LSTM layers in first block.
            lstm_block_two (int): Number of LSTM layers in second block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("CNN-LSTM")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(sub_seq, steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = cnnlstm(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            lstm_block_one=lstm_block_one,
            lstm_block_two=lstm_block_two,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3],
            ),
            output_shape=self.input_y.shape[1],
        )

    def create_cnngru(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        gru_block_one: int = 1,
        gru_block_two: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    ) -> None:
        """Creates CNN-GRU hybrid model.
        Parameters:
            sub_seq (int): Divide data into further subsequences.
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            conv_block_one (int): Number of convolutional layers in first block.
            conv_block_two (int): Number of convolutional layers in second block.
            gru_block_one (int): Number of GRU layers in first block.
            gru_block_two (int): Number of GRU layers in second block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("CNN-GRU")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(sub_seq, steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = cnngru(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            gru_block_one=gru_block_one,
            gru_block_two=gru_block_two,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3],
            ),
            output_shape=self.input_y.shape[1],
        )

    def create_cnnbirnn(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        birnn_block_one: int = 1,
        rnn_block_one: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    ) -> None:
        """Creates CNN-BI-RNN hybrid model.
        Parameters:
            sub_seq (int): Divide data into further subsequences.
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            conv_block_one (int): Number of convolutional layers in first block.
            conv_block_two (int): Number of convolutional layers in second block.
            birnn_block_one (int): Number of BI-RNN layers in first block.
            rnn_block_one (int): Number of RNN layers in second block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("CNN-BI-RNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(sub_seq, steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = cnnbirnn(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            birnn_block_one=birnn_block_one,
            rnn_block_one=rnn_block_one,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3],
            ),
            output_shape=self.input_y.shape[1],
        )

    def create_cnnbilstm(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        bilstm_block_one: int = 1,
        lstm_block_one: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    ) -> None:
        """Creates CNN-BI-LSTM hybrid model.
        Parameters:
            sub_seq (int): Divide data into further subsequences.
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            conv_block_one (int): Number of convolutional layers in first block.
            conv_block_two (int): Number of convolutional layers in second block.
            bilstm_block_one (int): Number of BI-LSTM layers in first block.
            lstm_block_one (int): Number of LSTM layers in second block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("CNN-BI-LSTM")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(sub_seq, steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = cnnbilstm(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            bilstm_block_one=bilstm_block_one,
            lstm_block_one=lstm_block_one,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3],
            ),
            output_shape=self.input_y.shape[1],
        )

    def create_cnnbigru(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        bigru_block_one: int = 1,
        gru_block_one: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    ) -> None:
        """Creates CNN-BI-GRU hybrid model.
        Parameters:
            sub_seq (int): Divide data into further subsequences.
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            optimizer (str): Optimization algorithm.
            optimizer_args (dict): Arguments for optimizer.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            conv_block_one (int): Number of convolutional layers in first block.
            conv_block_two (int): Number of convolutional layers in second block.
            bigru_block_one (int): Number of BI-GRU layers in first block.
            gru_block_one (int): Number of GRU layers in second block.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("CNN-BI-GRU")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self._model_intake_prep(sub_seq, steps_past, steps_future)

        optimizer_obj = get_optimizer(optimizer, optimizer_args)

        self.model = cnnbigru(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            bigru_block_one=bigru_block_one,
            gru_block_one=gru_block_one,
            layer_config=layer_config,
            input_shape=(
                self.input_x.shape[1],
                self.input_x.shape[2],
                self.input_x.shape[3],
            ),
            output_shape=self.input_y.shape[1],
        )

    def fit_model(
        self,
        epochs: int,
        show_progress: int = 1,
        board: bool = False,
        batch_size=None,
        **callback_setting: dict,
    ) -> Any:
        """Trains the model on data provided. Perfroms validation.
        Parameters:
            epochs (int): Number of epochs to train the model.
            show_progress (int): Prints training progress.
            board (bool): Creates TensorBoard.
            batch_size (float): Create Batch size.
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

    def evaluate_model(self, board: bool = False) -> Any:
        """Evaluate model on test set.
        Parameters:
            board (bool): Create TensorBoard.
        """
        if board == True:
            callback_board = TensorBoard(
                log_dir="logs/eval/"
                + self.model_id
                + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                histogram_freq=1,
            )
            self.evaluation_details = self.model.evaluate(
                x=self.input_x_test, y=self.input_y_test, callbacks=[callback_board]
            )
        else:
            self.evaluation_details = self.model.evaluate(
                x=self.input_x_test,
                y=self.input_y_test,
            )

        return self.evaluation_details

    def model_blueprint(self) -> Any:
        """Prints a summary of the models layer structure."""
        self.model.summary()

    def show_performance(self) -> Any:
        """Returns performance details."""
        return self.details

    def show_evaluation(self) -> Any:
        """Returns performance details on test data."""
        return self.evaluation_details

    def predict(
        self,
        data: np.ndarray,
        sub_seq: int = None,
        steps_past: int = None,
        steps_future: int = None,
    ) -> np.ndarray:
        """Takes in a sequence of values and outputs a forecast.
        Parameters:
            data (array): Input sequence which needs to be forecasted.
        Returns:
            (array): Forecast for sequence provided.
        """
        if sub_seq != None:
            self.sub_seq = sub_seq
        if steps_past != None:
            self.steps_past = steps_past
        if steps_future != None:
            self.steps_future = steps_future

        shape_ = int((data.shape[1] * self.steps_past) / self.sub_seq)
        data = data.reshape(1, self.sub_seq, shape_, 1)

        y_pred = self.model.predict(data, verbose=0)

        y_pred = y_pred.reshape(y_pred.shape[1], y_pred.shape[0])

        return y_pred

    def freeze(self, absolute_path: str = CURRENT_PATH) -> None:
        """Save the current model to the current directory.
        Parameters:
           absolute_path (str): Path to save model to.
        """
        self.model.save(absolute_path)

    def retrieve(self, location: str) -> None:
        """Load a keras model from the path specified.
        Parameters:
            location (str): Path of keras model location.
        """
        self.model = load_model(location)


class HybridMulti(BaseHybridMulti):
    def create_fit_cnnrnn(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        rnn_block_one: int = 1,
        rnn_block_two: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
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
    ) -> float:
        """Creates CNN-RNN hybrid model."""
        self.create_cnnrnn(
            sub_seq=sub_seq,
            steps_past=steps_past,
            steps_future=steps_future,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            rnn_block_one=rnn_block_one,
            rnn_block_two=rnn_block_two,
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

    def create_fit_cnnlstm(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        lstm_block_one: int = 1,
        lstm_block_two: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
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
    ) -> float:
        """Creates CNN-LSTM hybrid model."""
        self.create_cnnlstm(
            sub_seq=sub_seq,
            steps_past=steps_past,
            steps_future=steps_future,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            lstm_block_one=lstm_block_one,
            lstm_block_two=lstm_block_two,
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

    def create_fit_cnngru(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        gru_block_one: int = 1,
        gru_block_two: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
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
    ) -> float:
        """Creates CNN-GRU hybrid model."""
        self.create_cnngru(
            sub_seq=sub_seq,
            steps_past=steps_past,
            steps_future=steps_future,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
            gru_block_one=gru_block_one,
            gru_block_two=gru_block_two,
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

    def create_fit_cnnbirnn(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        birnn_block_one: int = 1,
        rnn_block_one: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
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
    ) -> float:
        """Creates CNN-BiRNN hybrid model."""
        self.create_cnnbirnn(
            sub_seq=sub_seq,
            steps_past=steps_past,
            steps_future=steps_future,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
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

    def create_fit_cnnbilstm(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        bilstm_block_one: int = 1,
        lstm_block_one: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
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
    ) -> float:
        """Creates CNN-BiLSTM hybrid model."""
        self.create_cnnbilstm(
            sub_seq=sub_seq,
            steps_past=steps_past,
            steps_future=steps_future,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
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

    def create_fit_cnnbigru(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        bigru_block_one: int = 1,
        gru_block_one: int = 1,
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
                    "dropout": 0.0,
                }
            },
            "layer4": {
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
    ) -> float:
        """Creates CNN-BiGRU hybrid model."""
        self.create_cnnbigru(
            sub_seq=sub_seq,
            steps_past=steps_past,
            steps_future=steps_future,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            loss=loss,
            metrics=metrics,
            conv_block_one=conv_block_one,
            conv_block_two=conv_block_two,
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


__all__ = ["HybridMulti"]
