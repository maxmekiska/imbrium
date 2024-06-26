from abc import ABC, abstractmethod

import numpy as np


class UniVariateMultiStep(ABC):
    """Abstract class that defines the general blueprint of a univariate
    multistep prediction object.
    """

    @abstractmethod
    def __init__(
        self,
        target: np.ndarray = np.array([]),
        features: np.ndarray = np.array([]),
        evaluation_split: float = 0.2,
        validation_split: float = 0.2,
    ):
        pass

    def _model_intake_prep(self, steps_past: int, steps_future: int) -> None:
        pass

    @abstractmethod
    def set_model_id(self, name: str):
        pass

    @property
    @abstractmethod
    def get_target(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def get_target_shape(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def get_model_id(self) -> str:
        pass

    @property
    @abstractmethod
    def get_X_input(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def get_X_input_shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def get_y_input(self) -> np.ndarray:
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
        board: bool = False,
        batch_size=None,
        show_progress: int = 1,
        validation_split: float = 0.20,
        **callback_setting: dict
    ):
        pass

    @abstractmethod
    def evaluate_model(self, board: bool = False):
        pass

    @abstractmethod
    def model_blueprint(self):
        pass

    @abstractmethod
    def show_performance(self):
        pass

    @abstractmethod
    def show_evaluation(self):
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def freeze(self, absolute_path: str):
        pass

    @abstractmethod
    def retrieve(self, location: str):
        pass
