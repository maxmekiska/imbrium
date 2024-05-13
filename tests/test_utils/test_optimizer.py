import keras
import pytest

from imbrium.utils.optimizer import get_optimizer


@pytest.mark.parametrize(
    "optimizer, optimizer_args, expected_optimizer_type",
    [
        ("adadelta", {"rho": 0.95}, keras.optimizers.Adadelta),
        ("adagrad", {"learning_rate": 0.01}, keras.optimizers.Adagrad),
        ("adam", {"learning_rate": 0.01}, keras.optimizers.Adam),
        ("adamax", {"learning_rate": 0.01}, keras.optimizers.Adamax),
        ("ftrl", {"learning_rate": 0.01}, keras.optimizers.Ftrl),
        ("nadam", {"learning_rate": 0.01}, keras.optimizers.Nadam),
        ("rmsprop", {"learning_rate": 0.01}, keras.optimizers.RMSprop),
        ("sgd", {"learning_rate": 0.01}, keras.optimizers.SGD),
    ],
)
def test_get_optimizer(optimizer, optimizer_args, expected_optimizer_type):
    optimizer_obj = get_optimizer(optimizer, optimizer_args)
    assert isinstance(optimizer_obj, expected_optimizer_type)


@pytest.mark.parametrize(
    "optimizer, optimizer_args, expected_optimizer_name",
    [
        ("adadelta", None, "adadelta"),
        ("adagrad", None, "adagrad"),
        ("adam", None, "adam"),
        ("adamax", None, "adamax"),
        ("ftrl", None, "ftrl"),
        ("nadam", None, "nadam"),
        ("rmsprop", None, "rmsprop"),
        ("sgd", None, "sgd"),
    ],
)
def test_get_optimizer_defualt(optimizer, optimizer_args, expected_optimizer_name):
    optimizer_name = get_optimizer(optimizer, optimizer_args)
    assert optimizer_name == expected_optimizer_name
