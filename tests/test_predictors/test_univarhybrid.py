import numpy as np
import pandas as pd
import pytest

from imbrium.predictors.univarhybrid import HybridUni

data = pd.read_csv("tests/example_dataset/CaliforniaHousing.csv")
data = data["target"]
data_small = data[:20]

test0 = HybridUni(2, 10, 3, data=data, scale="standard")
test1 = HybridUni(1, 5, 1, data=data, scale="standard")
test2 = HybridUni(5, 10, 2, data=data_small, scale="standard")

test0.create_cnnlstm(
    optimizer="adam",
    loss="mean_squared_error",
    metrics="mean_squared_error",
    layer_config={
        "layer0": (64, 1, "relu", 0.0, 0.0),
        "layer1": (32, 1, "relu", 0.0, 0.0),
        "layer2": (2),
        "layer3": (50, "relu", 0.0, 0.0),
        "layer4": (25, "relu", 0.0),
    },
)

X = np.array(
    [
        [[1.17289952], [0.54461086], [0.80025935], [0.29936163], [0.17197069]],
        [[0.47008283], [0.64687025], [0.30282805], [0.05757883], [-0.13480749]],
    ]
)

y = np.array([[0.05757883], [-0.13480749], [-0.41298771]])


shape_x = (20628, 2, 5, 1)
shape_y = (20628, 3, 1)

model_id = "CNN-LSTM"
optimizer = "adam"
loss = "mean_squared_error"
metrics = "mean_squared_error"


def test_get_model_id():
    assert test0.get_model_id == model_id


def test_get_X_input():
    np.testing.assert_allclose(test0.get_X_input[4], X)


def test_get_y_input():
    np.testing.assert_allclose(test0.get_y_input[2], y)


def test_get_X_input_shape():
    np.testing.assert_allclose(test0.get_X_input_shape, shape_x)


def test_get_y_input_shape():
    np.testing.assert_allclose(test0.get_y_input_shape, shape_y)


def test_get_optimizer():
    assert test0.get_optimizer == optimizer


def test_get_loss():
    assert test0.get_loss == loss


def test_get_metrics():
    assert test0.get_metrics == metrics


def test_create_cnnrnn():
    test0.create_cnnrnn(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
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
    )
    assert test0.get_model_id == "CNN-RNN"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnnlstm():
    test0.create_cnnlstm(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "CNN-LSTM"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnngru():
    test0.create_cnngru(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "CNN-GRU"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnnbirnn():
    test0.create_cnnbirnn(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "CNN-BI-RNN"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnnbilstm():
    test0.create_cnnbilstm(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "CNN-BI-LSTM"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnnbigru():
    test0.create_cnnbigru(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (64, 1, "relu", 0.0, 0.0),
            "layer1": (32, 1, "relu", 0.0, 0.0),
            "layer2": (2),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (25, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "CNN-BI-GRU"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_fit_cnnrnn():
    try:
        test2.create_fit_cnnrnn(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnlstm():
    try:
        test2.create_fit_cnnlstm(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnngru():
    try:
        test2.create_fit_cnngru(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbirnn():
    try:
        test2.create_fit_cnnbirnn(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbilstm():
    try:
        test2.create_fit_cnnbilstm(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbigru():
    try:
        test2.create_fit_cnnbigru(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")
