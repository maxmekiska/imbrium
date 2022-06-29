from abc import ABC
from abc import abstractmethod
from pandas import DataFrame
from numpy import array

class MultiVariateMultiStep(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _scaling(self, method: str) -> object:
        pass

    @abstractmethod
    def _data_prep(self, data: DataFrame, features: list) -> array:
        pass

    @abstractmethod
    def _sequence_prep(self, input_sequence: array, steps_past: int, steps_future: int) -> [(array, array)]:
        pass

    @abstractmethod
    def _multistep_prep(self, input_sequence: array, steps_past: int, steps_future: int) -> [(array, array)]:
        pass

    @property
    @abstractmethod
    def get_X_input(self) -> array:
        pass

    @property
    @abstractmethod
    def get_X_input_shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def get_y_input(self) -> array:
        pass

    @property
    @abstractmethod
    def get_y_input_shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def get_loss(self) -> str:
        pass

    @property
    @abstractmethod
    def get_metrics(self) -> str:
        pass

    @abstractmethod
    def fit_model(self, epochs: int, show_progress: int = 1, validation_split: float = 0.20, batch_size: int = 10):
        pass

    @abstractmethod
    def model_blueprint(self):
        pass

    @abstractmethod
    def show_performance(self):
        pass

    @abstractmethod
    def predict(self, data: array) -> DataFrame:
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self, location: str):
        pass
