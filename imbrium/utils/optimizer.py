from typing import Dict, Optional, Type, Union

import keras
from keras.optimizers import (SGD, Adadelta, Adagrad, Adam, Adamax, Ftrl,
                              Nadam, Optimizer, RMSprop)


def get_optimizer(
    optimizer: str, optimizer_args: Optional[Dict[str, Union[int, float, str]]] = None
) -> Union[str, Optimizer]:
    """Get optimizer object from string name and arguments."""
    optimizer_dict: Dict[str, Type[Optimizer]] = {
        "adadelta": Adadelta,
        "adagrad": Adagrad,
        "adam": Adam,
        "adamax": Adamax,
        "ftrl": Ftrl,
        "nadam": Nadam,
        "rmsprop": RMSprop,
        "sgd": SGD,
    }

    if optimizer_args is None:
        return optimizer
    else:
        return optimizer_dict[optimizer](**optimizer_args)
