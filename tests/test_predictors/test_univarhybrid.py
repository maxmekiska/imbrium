import numpy as np
import pandas as pd
import pytest

from imbrium.predictors.univarhybrid import HybridUni

data = pd.read_csv("tests/example_dataset/mockData.csv")
data = np.array(data["target"])
data_small = data[:20]

test0 = HybridUni(target=data)
test2 = HybridUni(target=data_small, validation_split=0.2, evaluation_split=0.2)

test0.create_cnnlstm(
    sub_seq=2,
    steps_past=10,
    steps_future=3,
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

X = np.array(
    [
        [[3.422], [2.697], [2.992], [2.414], [2.267]],
        [[2.611], [2.815], [2.418], [2.135], [1.913]],
    ]
)

y = np.array([[2.135], [1.913], [1.592]])


shape_x = (438, 2, 5, 1)
shape_y = (438, 3, 1)

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
        sub_seq=2,
        steps_past=10,
        steps_future=3,
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
        sub_seq=2,
        steps_past=10,
        steps_future=3,
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
    assert test0.get_model_id == "CNN-LSTM"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnngru():
    test0.create_cnngru(
        sub_seq=2,
        steps_past=10,
        steps_future=3,
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
    assert test0.get_model_id == "CNN-GRU"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnnbirnn():
    test0.create_cnnbirnn(
        sub_seq=2,
        steps_past=10,
        steps_future=3,
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
    assert test0.get_model_id == "CNN-BI-RNN"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnnbilstm():
    test0.create_cnnbilstm(
        sub_seq=2,
        steps_past=10,
        steps_future=3,
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
    assert test0.get_model_id == "CNN-BI-LSTM"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnnbigru():
    test0.create_cnnbigru(
        sub_seq=2,
        steps_past=10,
        steps_future=3,
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
    assert test0.get_model_id == "CNN-BI-GRU"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_fit_cnnrnn_evaluate():
    try:
        test2.create_fit_cnnrnn(sub_seq=5, steps_past=10, steps_future=2, epochs=1)
        test2.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnlstm_evaluate():
    try:
        test2.create_fit_cnnlstm(sub_seq=5, steps_past=10, steps_future=2, epochs=1)
        test2.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnngru_evalaute():
    try:
        test2.create_fit_cnngru(sub_seq=5, steps_past=10, steps_future=2, epochs=1)
        test2.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbirnn_evaluate():
    try:
        test2.create_fit_cnnbirnn(sub_seq=5, steps_past=10, steps_future=2, epochs=1)
        test2.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbilstm_evaluate():
    try:
        test2.create_fit_cnnbilstm(sub_seq=5, steps_past=10, steps_future=2, epochs=1)
        test2.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbigru_evaluate():
    try:
        test2.create_fit_cnnbigru(sub_seq=5, steps_past=10, steps_future=2, epochs=1)
        test2.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")
