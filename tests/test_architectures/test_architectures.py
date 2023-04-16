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
                dense_block_one=1,
                dense_block_two=1,
                dense_block_three=1,
                layer_config={
                    "layer0": (50, "relu", 0.0, 0.0),
                    "layer1": (25, "relu", 0.0, 0.0),
                    "layer2": (25, "relu", 0.0),
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
                rnn_block_one=1,
                rnn_block_two=1,
                rnn_block_three=1,
                layer_config={
                    "layer0": (50, "relu", 0.0, 0.0),
                    "layer1": (25, "relu", 0.0, 0.0),
                    "layer2": (25, "relu", 0.0),
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
                lstm_block_one=1,
                lstm_block_two=1,
                lstm_block_three=1,
                layer_config={
                    "layer0": (50, "relu", 0.0, 0.0),
                    "layer1": (25, "relu", 0.0, 0.0),
                    "layer2": (25, "relu", 0.0),
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
                conv_block_one=1,
                conv_block_two=1,
                dense_block_one=1,
                layer_config={
                    "layer0": (64, 1, "relu", 0.0, 0.0),
                    "layer1": (32, 1, "relu", 0.0, 0.0),
                    "layer2": (2),
                    "layer3": (50, "relu", 0.0),
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
                gru_block_one=1,
                gru_block_two=1,
                gru_block_three=1,
                layer_config={
                    "layer0": (50, "relu", 0.0, 0.0),
                    "layer1": (25, "relu", 0.0, 0.0),
                    "layer2": (25, "relu", 0.0),
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
                birnn_block_one=1,
                rnn_block_one=1,
                layer_config={
                    "layer0": (50, "relu", 0.0, 0.0),
                    "layer1": (50, "relu", 0.0),
                },
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
                bilstm_block_one=1,
                lstm_block_one=1,
                layer_config={
                    "layer0": (50, "relu", 0.0, 0.0),
                    "layer1": (50, "relu", 0.0),
                },
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
                bigru_block_one=1,
                gru_block_one=1,
                layer_config={
                    "layer0": (50, "relu", 0.0, 0.0),
                    "layer1": (50, "relu", 0.0),
                },
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
                enc_rnn_block_one=1,
                enc_rnn_block_two=1,
                dec_rnn_block_one=1,
                dec_rnn_block_two=1,
                layer_config={
                    "layer0": (100, "relu", 0.0, 0.0),
                    "layer1": (50, "relu", 0.0, 0.0),
                    "layer2": (50, "relu", 0.0, 0.0),
                    "layer3": (100, "relu", 0.0),
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
                enc_lstm_block_one=1,
                enc_lstm_block_two=1,
                dec_lstm_block_one=1,
                dec_lstm_block_two=1,
                layer_config={
                    "layer0": (100, "relu", 0.0, 0.0),
                    "layer1": (50, "relu", 0.0, 0.0),
                    "layer2": (50, "relu", 0.0, 0.0),
                    "layer3": (100, "relu", 0.0),
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
                enc_conv_block_one=1,
                enc_conv_block_two=1,
                dec_gru_block_one=1,
                dec_gru_block_two=1,
                metrics="mean_squared_error",
                layer_config={
                    "layer0": (64, 1, "relu", 0.0, 0.0),
                    "layer1": (32, 1, "relu", 0.0, 0.0),
                    "layer2": (2),
                    "layer3": (50, "relu", 0.0, 0.0),
                    "layer4": (100, "relu", 0.0),
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
                enc_gru_block_one=1,
                enc_gru_block_two=1,
                dec_gru_block_one=1,
                dec_gru_block_two=1,
                layer_config={
                    "layer0": (100, "relu", 0.0, 0.0),
                    "layer1": (50, "relu", 0.0, 0.0),
                    "layer2": (50, "relu", 0.0, 0.0),
                    "layer3": (100, "relu", 0.0),
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
                conv_block_one=1,
                conv_block_two=1,
                rnn_block_one=1,
                rnn_block_two=1,
                layer_config={
                    "layer0": (64, 1, "relu", 0.0, 0.0),
                    "layer1": (32, 1, "relu", 0.0, 0.0),
                    "layer2": (2),
                    "layer3": (50, "relu", 0.0, 0.0),
                    "layer4": (25, "relu", 0.0),
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
                    "layer0": (64, 1, "relu", 0.0, 0.0),
                    "layer1": (32, 1, "relu", 0.0, 0.0),
                    "layer2": (2),
                    "layer3": (50, "relu", 0.0, 0.0),
                    "layer4": (25, "relu", 0.0),
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
                    "layer0": (64, 1, "relu", 0.0, 0.0),
                    "layer1": (32, 1, "relu", 0.0, 0.0),
                    "layer2": (2),
                    "layer3": (50, "relu", 0.0, 0.0),
                    "layer4": (25, "relu", 0.0),
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
                    "layer0": (64, 1, "relu", 0.0, 0.0),
                    "layer1": (32, 1, "relu", 0.0, 0.0),
                    "layer2": (2),
                    "layer3": (50, "relu", 0.0, 0.0),
                    "layer4": (25, "relu", 0.0),
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
                    "layer0": (64, 1, "relu", 0.0, 0.0),
                    "layer1": (32, 1, "relu", 0.0, 0.0),
                    "layer2": (2),
                    "layer3": (50, "relu", 0.0, 0.0),
                    "layer4": (25, "relu", 0.0),
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
                    "layer0": (64, 1, "relu", 0.0, 0.0),
                    "layer1": (32, 1, "relu", 0.0, 0.0),
                    "layer2": (2),
                    "layer3": (50, "relu", 0.0, 0.0),
                    "layer4": (25, "relu", 0.0),
                },
                input_shape=(3, 3, 3),
                output_shape=3,
            )
        )
        == keras_obj
    )
