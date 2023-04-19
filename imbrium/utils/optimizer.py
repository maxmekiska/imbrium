from tensorflow import keras


def get_optimizer(optimizer: str, optimizer_args: dict) -> keras.optimizers.Optimizer:
    """Get optimizer object from string name and arguments."""
    optimizer_dict = {
        "adadelta": keras.optimizers.Adadelta,
        "adagrad": keras.optimizers.Adagrad,
        "adam": keras.optimizers.Adam,
        "adamax": keras.optimizers.Adamax,
        "ftrl": keras.optimizers.Ftrl,
        "nadam": keras.optimizers.Nadam,
        "rmsprop": keras.optimizers.RMSprop,
        "sgd": keras.optimizers.SGD,
    }

    if optimizer_args is None:
        optimizer_obj = optimizer
    else:
        optimizer_obj = optimizer_dict.get(optimizer)(**optimizer_args)

    return optimizer_obj
