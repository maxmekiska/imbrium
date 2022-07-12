import unittest

from imbrium.architectures.models import *

class TestModels(unittest.TestCase):

    keras_obj = type(tf.keras.Sequential())

    def test_mlp(self):
        self.assertEqual(type(mlp(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'mean_squared_error', layer_config = {'layer0': (50, 'relu'), 'layer1': (25,'relu'), 'layer2': (25, 'relu')}, input_shape = 3 , output_shape = 3)), TestModels.keras_obj)

    def test_rnn(self):
        self.assertEqual(type(rnn(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'mean_squared_error', layer_config = {'layer0': (50, 'relu'), 'layer1': (25,'relu'), 'layer2': (25, 'relu')}, input_shape = (3, 3) , output_shape = 3)), TestModels.keras_obj)

    def test_lstm(self):
        self.assertEqual(type(lstm(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'mean_squared_error', layer_config = {'layer0': (50, 'relu'), 'layer1': (25,'relu'), 'layer2': (25, 'relu')}, input_shape = (3, 3) , output_shape = 3)), TestModels.keras_obj)

    def test_cnn(self):
        self.assertEqual(type(cnn(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'mean_squared_error', layer_config = {'layer0': (64, 1, 'relu'), 'layer1': (32, 1, 'relu'), 'layer2': (2), 'layer3': (50, 'relu')}, input_shape = (3, 3) , output_shape = 3)), TestModels.keras_obj)

    def test_gru(self):
        self.assertEqual(type(gru(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'mean_squared_error', layer_config = {'layer0': (50, 'relu'), 'layer1': (25,'relu'), 'layer2': (25, 'relu')}, input_shape = (3, 3) , output_shape = 3)), TestModels.keras_obj)

    def test_birnn(self):
        self.assertEqual(type(birnn(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'mean_squared_error', layer_config = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')}, input_shape = (3, 3) , output_shape = 3)), TestModels.keras_obj)

    def test_bilstm(self):
        self.assertEqual(type(bilstm(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'mean_squared_error', layer_config = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')}, input_shape = (3, 3) , output_shape = 3)), TestModels.keras_obj)

    def test_bigru(self):
        self.assertEqual(type(bigru(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'mean_squared_error', layer_config = {'layer0': (50, 'relu'), 'layer1': (50, 'relu')}, input_shape = (3, 3) , output_shape = 3)), TestModels.keras_obj)
