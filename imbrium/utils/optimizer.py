import keras_core


def get_optimizer(
    optimizer: str, optimizer_args: dict
) -> keras_core.optimizers.Optimizer:
    """Get optimizer object from string name and arguments."""
    optimizer_dict = {
        "adadelta": keras_core.optimizers.Adadelta,
        "adagrad": keras_core.optimizers.Adagrad,
        "adam": keras_core.optimizers.Adam,
        "adamax": keras_core.optimizers.Adamax,
        "ftrl": keras_core.optimizers.Ftrl,
        "nadam": keras_core.optimizers.Nadam,
        "rmsprop": keras_core.optimizers.RMSprop,
        "sgd": keras_core.optimizers.SGD,
    }

    if optimizer_args is None:
        optimizer_obj = optimizer
    else:
        optimizer_obj = optimizer_dict.get(optimizer)(**optimizer_args)

    return optimizer_obj
