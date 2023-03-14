import numpy as np
import pandas as pd
import pytest

from imbrium.predictors.univarhybrid import *

data = pd.read_csv("tests/example_dataset/CaliforniaHousing.csv")
data = data["target"]

test0 = HybridUni(2, 10, 3, data=data, scale="standard")

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
