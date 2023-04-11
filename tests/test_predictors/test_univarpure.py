import numpy as np
import pandas as pd
import pytest

from imbrium.predictors.univarpure import *

data = pd.read_csv("tests/example_dataset/CaliforniaHousing.csv")
data = data["target"]
data_small = data[:20]

test0 = PureUni(2, 3, data=data, scale="standard")
test1 = PureUni(1, 5, data=data, scale="standard")
test2 = OptimizePureUni(5, 10, data=data_small, scale="standard")
test3 = OptimizePureUni(5, 5, data=data_small, scale="standard")

test0.create_lstm(
    optimizer="adam",
    loss="mean_squared_error",
    metrics="mean_squared_error",
    layer_config={
        "layer0": (40, "relu", 0.0, 0.0),
        "layer1": (50, "relu", 0.0, 0.0),
        "layer2": (50, "relu", 0.0),
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


def test_create_mlp():
    test0.create_mlp(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (40, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0, 0.0),
            "layer2": (50, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "MLP"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_rnn():
    test0.create_rnn(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (40, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0, 0.0),
            "layer2": (50, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "RNN"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnn():
    test0.create_cnn(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (40, 1, "relu", 0.0, 0.0),
            "layer1": (50, 1, "relu", 0.0, 0.0),
            "layer2": (1),
            "layer3": (50, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "CNN"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_gru():
    test0.create_gru(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (40, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0, 0.0),
            "layer2": (50, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "GRU"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_birnn():
    test0.create_birnn(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (40, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0, 0.0),
            "layer2": (50, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "BI-RNN"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_bilstm():
    test0.create_bilstm(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (40, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0, 0.0),
            "layer2": (50, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "BI-LSTM"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_bigru():
    test0.create_bigru(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (40, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0, 0.0),
            "layer2": (50, "relu", 0.0),
        },
    )
    assert test0.get_model_id == "BI-GRU"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_encdec_rnn():
    test1.create_encdec_rnn(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (40, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0, 0.0),
            "layer2": (50, "relu", 0.0, 0.0),
            "layer3": (50, "relu", 0.0),
        },
    )
    assert test1.get_model_id == "Encoder-Decoder-RNN"
    assert test1.get_optimizer == "adam"
    assert test1.get_loss == "mean_squared_error"
    assert test1.get_metrics == "mean_squared_error"


def test_create_encdec_lstm():
    test1.create_encdec_lstm(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (40, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0, 0.0),
            "layer2": (50, "relu", 0.0, 0.0),
            "layer3": (50, "relu", 0.0),
        },
    )
    assert test1.get_model_id == "Encoder-Decoder-LSTM"
    assert test1.get_optimizer == "adam"
    assert test1.get_loss == "mean_squared_error"
    assert test1.get_metrics == "mean_squared_error"


def test_create_encdec_gru():
    test1.create_encdec_gru(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (40, "relu", 0.0, 0.0),
            "layer1": (50, "relu", 0.0, 0.0),
            "layer2": (50, "relu", 0.0, 0.0),
            "layer3": (50, "relu", 0.0),
        },
    )
    assert test1.get_model_id == "Encoder-Decoder-GRU"
    assert test1.get_optimizer == "adam"
    assert test1.get_loss == "mean_squared_error"
    assert test1.get_metrics == "mean_squared_error"


def test_create_encdec_cnn():
    test1.create_encdec_cnn(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": (40, 1, "relu", 0.0, 0.0),
            "layer1": (50, 1, "relu", 0.0, 0.0),
            "layer2": (1),
            "layer3": (50, "relu", 0.0, 0.0),
            "layer4": (50, "relu", 0.0, 0.0),
            "layer5": (1),
            "layer6": (50, "relu", 0.0),
        },
    )
    assert test1.get_model_id == "Encoder(CNN)-Decoder(GRU)"
    assert test1.get_optimizer == "adam"
    assert test1.get_loss == "mean_squared_error"
    assert test1.get_metrics == "mean_squared_error"


def test_create_fit_mlp():
    try:
        test2.create_fit_mlp(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_rnn():
    try:
        test2.create_fit_rnn(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_lstm():
    try:
        test2.create_fit_lstm(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_gru():
    try:
        test2.create_fit_gru(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnn():
    try:
        test2.create_fit_cnn(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_encdec_rnn():
    try:
        test3.create_fit_encdec_rnn(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_encdec_lstm():
    try:
        test3.create_fit_encdec_lstm(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_encdec_gru():
    try:
        test3.create_fit_encdec_gru(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_encdec_cnn():
    try:
        test3.create_fit_encdec_cnn(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")
