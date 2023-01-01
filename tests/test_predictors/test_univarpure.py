import numpy as np
import pandas as pd
import pytest

from imbrium.predictors.univarpure import *

data = pd.read_csv("tests/example_dataset/CaliforniaHousing.csv")
data = data["target"]

test0 = PureUni(2, 3, data=data, scale="standard")

test0.create_lstm(
    optimizer="adam",
    loss="mean_squared_error",
    metrics="mean_squared_error",
    layer_config={
        "layer0": (40, "relu"),
        "layer1": (50, "relu"),
        "layer2": (50, "relu"),
    },
)

X = np.array([[1.17289952], [0.54461086]])

y = np.array([[1.17289952], [0.54461086], [0.80025935]])

shape_x = (20636, 2, 1)
shape_y = (20636, 3, 1)

model_id = "LSTM"
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
