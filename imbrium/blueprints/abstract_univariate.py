from abc import ABC, abstractmethod

from numpy import array
from pandas import DataFrame


class UniVariateMultiStep(ABC):
    """Abstract class that defines the general blueprint of a univariate
    multistep prediction object.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def set_model_id(self, name: str):
        pass

    @property
    @abstractmethod
    def get_model_id(self) -> str:
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
    def get_optimizer(self) -> str:
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
    def fit_model(
        self,
        epochs: int,
        show_progress: int = 1,
        validation_split: float = 0.20,
        batch_size: int = 10,
        **callback_setting: dict
    ):
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
    def save_model(self, absolute_path: str):
        pass

    @abstractmethod
    def load_model(self, location: str):
        pass
