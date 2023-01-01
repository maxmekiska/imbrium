import pytest

from imbrium.architectures.models import *

keras_obj = type(tf.keras.Sequential())


def test_mlp():
    assert (
        type(
            mlp(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (50, "relu"),
                    "layer1": (25, "relu"),
                    "layer2": (25, "relu"),
                },
                input_shape=3,
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_rnn():
    assert (
        type(
            rnn(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (50, "relu"),
                    "layer1": (25, "relu"),
                    "layer2": (25, "relu"),
                },
                input_shape=(3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_lstm():
    assert (
        type(
            lstm(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (50, "relu"),
                    "layer1": (25, "relu"),
                    "layer2": (25, "relu"),
                },
                input_shape=(3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_cnn():
    assert (
        type(
            cnn(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (64, 1, "relu"),
                    "layer1": (32, 1, "relu"),
                    "layer2": (2),
                    "layer3": (50, "relu"),
                },
                input_shape=(3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_gru():
    assert (
        type(
            gru(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (50, "relu"),
                    "layer1": (25, "relu"),
                    "layer2": (25, "relu"),
                },
                input_shape=(3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_birnn():
    assert (
        type(
            birnn(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={"layer0": (50, "relu"), "layer1": (50, "relu")},
                input_shape=(3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_bilstm():
    assert (
        type(
            bilstm(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={"layer0": (50, "relu"), "layer1": (50, "relu")},
                input_shape=(3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_bigru():
    assert (
        type(
            bigru(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={"layer0": (50, "relu"), "layer1": (50, "relu")},
                input_shape=(3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_encdec_rnn():
    assert (
        type(
            encdec_rnn(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (100, "relu"),
                    "layer1": (50, "relu"),
                    "layer2": (50, "relu"),
                    "layer3": (100, "relu"),
                },
                input_shape=(3, 3),
                output_shape=3,
                repeat=3,
            )
        )
        == keras_obj
    )


def test_encdec_lstm():
    assert (
        type(
            encdec_lstm(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (100, "relu"),
                    "layer1": (50, "relu"),
                    "layer2": (50, "relu"),
                    "layer3": (100, "relu"),
                },
                input_shape=(3, 3),
                output_shape=3,
                repeat=3,
            )
        )
        == keras_obj
    )


def test_encdec_cnn():
    assert (
        type(
            encdec_cnn(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (64, 1, "relu"),
                    "layer1": (32, 1, "relu"),
                    "layer2": (2),
                    "layer3": (50, "relu"),
                    "layer4": (100, "relu"),
                },
                input_shape=(3, 3),
                output_shape=3,
                repeat=3,
            )
        )
        == keras_obj
    )


def test_encdec_gru():
    assert (
        type(
            encdec_gru(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (100, "relu"),
                    "layer1": (50, "relu"),
                    "layer2": (50, "relu"),
                    "layer3": (100, "relu"),
                },
                input_shape=(3, 3),
                output_shape=3,
                repeat=3,
            )
        )
        == keras_obj
    )


def test_cnnrnn():
    assert (
        type(
            cnnrnn(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (64, 1, "relu"),
                    "layer1": (32, 1, "relu"),
                    "layer2": (2),
                    "layer3": (50, "relu"),
                    "layer4": (25, "relu"),
                },
                input_shape=(3, 3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_cnnlstm():
    assert (
        type(
            cnnlstm(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (64, 1, "relu"),
                    "layer1": (32, 1, "relu"),
                    "layer2": (2),
                    "layer3": (50, "relu"),
                    "layer4": (25, "relu"),
                },
                input_shape=(3, 3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_cnngru():
    assert (
        type(
            cnngru(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (64, 1, "relu"),
                    "layer1": (32, 1, "relu"),
                    "layer2": (2),
                    "layer3": (50, "relu"),
                    "layer4": (25, "relu"),
                },
                input_shape=(3, 3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_cnnbirnn():
    assert (
        type(
            cnnbirnn(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (64, 1, "relu"),
                    "layer1": (32, 1, "relu"),
                    "layer2": (2),
                    "layer3": (50, "relu"),
                    "layer4": (25, "relu"),
                },
                input_shape=(3, 3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_cnnbilstm():
    assert (
        type(
            cnnbilstm(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (64, 1, "relu"),
                    "layer1": (32, 1, "relu"),
                    "layer2": (2),
                    "layer3": (50, "relu"),
                    "layer4": (25, "relu"),
                },
                input_shape=(3, 3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )


def test_cnnbigru():
    assert (
        type(
            cnnbigru(
                optimizer="adam",
                loss="mean_squared_error",
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (64, 1, "relu"),
                    "layer1": (32, 1, "relu"),
                    "layer2": (2),
                    "layer3": (50, "relu"),
                    "layer4": (25, "relu"),
                },
                input_shape=(3, 3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )
