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
from imbrium.utils.transformer import data_prep_multi, multistep_prep_hybrid


class BaseHybridMulti(MultiVariateMultiStep):
    """Implements neural network based multivariate multipstep hybrid predictors."""


    CURRENT_PATH = os.getcwd()

    def __init__(
        self,
        sub_seq: int,
        steps_past: int,
        steps_future: int,
        data=DataFrame(),
        features: list = [],
        scale: str = "",
    ) -> object:
        """
        Parameters:
            sub_seq (int): Divide data into further subsequences.
            steps_past (int): Steps predictor will look backward.
            steps_future (int): Steps predictor will look forward.
            data (DataFrame): Input data for model training.
            features (list): List of features. First feature in list will be
            set to the target variable.
            scale (str): How to scale the data before making predictions.
        """
        self.sub_seq = sub_seq
        self.steps_past = steps_past
        self.steps_future = steps_future
        self.optimizer = ""
        self.loss = ""
        self.metrics = ""

        self.scaler = SCALER[scale]

        self.model_id = ""
        self.sub_seq = sub_seq

        if len(data) > 0:
            self.data = data_prep_multi(data, features, self.scaler)
            self.input_x, self.input_y, self.modified_back = multistep_prep_hybrid(
                self.data, sub_seq, steps_past, steps_future
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

    def create_cnnrnn(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one=1,
        conv_block_two=1,
        rnn_block_one=1,
        rnn_block_two=1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
    ):
        """Creates CNN-RNN hybrid model.
        Parameters:
            optimizer (str): Optimization algorithm.
            loss (str): Loss function.
            metrics (str): Performance measurement.
            layer_config (dict): Adjust neurons and acitivation functions.
        """
        self.set_model_id("CNN-RNN")
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

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
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one=1,
        conv_block_two=1,
        lstm_block_one=1,
        lstm_block_two=1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
    ):
        """Creates CNN-LSTM hybrid model.
        Parameters:
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
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one=1,
        conv_block_two=1,
        gru_block_one=1,
        gru_block_two=1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
    ):
        """Creates CNN-GRU hybrid model.
        Parameters:
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
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one=1,
        conv_block_two=1,
        birnn_block_one=1,
        rnn_block_one=1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
    ):
        """Creates CNN-BI-RNN hybrid model.
        Parameters:
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
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one=1,
        conv_block_two=1,
        bilstm_block_one=1,
        lstm_block_one=1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
    ):
        """Creates CNN-BI-LSTM hybrid model.
        Parameters:
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
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one=1,
        conv_block_two=1,
        bigru_block_one=1,
        gru_block_one=1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
    ):
        """Creates CNN-BI-GRU hybrid model.
        Parameters:
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
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Trains the model on data provided. Perfroms validation.
        Parameters:
            epochs (int): Number of epochs to train the model.
            show_progress (int): Prints training progress.
            validation_split (float): Determines size of Validation data.
            board (bool): Creates TensorBoard.
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

        shape_ = int((data.shape[1] * self.steps_past) / self.sub_seq)
        data = data.reshape(1, self.sub_seq, shape_, 1)

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


class HybridMulti(BaseHybridMulti):
    def create_fit_cnnrnn(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        rnn_block_one: int = 1,
        rnn_block_two: int = 1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates CNN-RNN hybrid model."""
        self.create_cnnrnn(
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
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_cnnlstm(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        lstm_block_one: int = 1,
        lstm_block_two: int = 1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates CNN-LSTM hybrid model."""
        self.create_cnnlstm(
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
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_cnngru(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        gru_block_one: int = 1,
        gru_block_two: int = 1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates CNN-GRU hybrid model."""
        self.create_cnngru(
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
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_cnnbirnn(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        birnn_block_one: int = 1,
        rnn_block_one: int = 1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates CNN-BiRNN hybrid model."""
        self.create_cnnbirnn(
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
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_cnnbilstm(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        bilstm_block_one: int = 1,
        lstm_block_one: int = 1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates CNN-BiLSTM hybrid model."""
        self.create_cnnbilstm(
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
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]

    def create_fit_cnnbigru(
        self,
        optimizer: str = "adam",
        optimizer_args: dict = None,
        loss: str = "mean_squared_error",
        metrics: str = "mean_squared_error",
        conv_block_one: int = 1,
        conv_block_two: int = 1,
        bigru_block_one: int = 1,
        gru_block_one: int = 1,
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
        epochs: int = 100,
        show_progress: int = 1,
        validation_split: float = 0.20,
        board: bool = False,
        **callback_setting: dict,
    ):
        """Creates CNN-BiGRU hybrid model."""
        self.create_cnnbigru(
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
            validation_split=validation_split,
            board=board,
            **callback_setting,
        )
        return self.details.history[metrics][-1]


__all__ = ["HybridMulti"]
