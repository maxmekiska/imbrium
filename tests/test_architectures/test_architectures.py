import unittest

from imbrium.architectures.models import *


class TestModels(unittest.TestCase):

    keras_obj = type(tf.keras.Sequential())

    def test_mlp(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_rnn(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_lstm(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_cnn(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_gru(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_birnn(self):
        self.assertEqual(
            type(
                birnn(
                    optimizer="adam",
                    loss="mean_squared_error",
                    metrics="mean_squared_error",
                    layer_config={"layer0": (50, "relu"), "layer1": (50, "relu")},
                    input_shape=(3, 3),
                    output_shape=3,
                )
            ),
            TestModels.keras_obj,
        )

    def test_bilstm(self):
        self.assertEqual(
            type(
                bilstm(
                    optimizer="adam",
                    loss="mean_squared_error",
                    metrics="mean_squared_error",
                    layer_config={"layer0": (50, "relu"), "layer1": (50, "relu")},
                    input_shape=(3, 3),
                    output_shape=3,
                )
            ),
            TestModels.keras_obj,
        )

    def test_bigru(self):
        self.assertEqual(
            type(
                bigru(
                    optimizer="adam",
                    loss="mean_squared_error",
                    metrics="mean_squared_error",
                    layer_config={"layer0": (50, "relu"), "layer1": (50, "relu")},
                    input_shape=(3, 3),
                    output_shape=3,
                )
            ),
            TestModels.keras_obj,
        )

    def test_encdec_rnn(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_encdec_lstm(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_encdec_cnn(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_encdec_gru(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_cnnrnn(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_cnnlstm(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_cnngru(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_cnnbirnn(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_cnnbilstm(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )

    def test_cnnbigru(self):
        self.assertEqual(
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
            ),
            TestModels.keras_obj,
        )


if __name__ == "__main__":
    unittest.main()
