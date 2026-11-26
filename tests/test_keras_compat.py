"""Tests for loading the U-Net consistently across Keras backends."""

import os
import tempfile
import unittest

import keras
import numpy as np

import segmenteveryforam as sef
from segmenteveryforam.keras_compat import SamePaddingConv2DTranspose, make_backend_consistent


def random_unet(seed=0):
    model = sef.Unet()
    rng = np.random.default_rng(seed)
    model.set_weights([
        (rng.normal(0, 0.1, w.shape) if w.ndim > 1 else w + rng.normal(0, 0.1, w.shape)).astype("float32")
        for w in model.get_weights()
    ])
    return model


class TestSamePaddingConv2DTranspose(unittest.TestCase):

    def test_matches_same_padding_layer(self):
        x = np.random.default_rng(1).random((2, 16, 16, 8)).astype("float32")
        original = keras.layers.Conv2DTranspose(4, 3, strides=2, padding="same")
        expected = np.asarray(original(x))
        replacement = SamePaddingConv2DTranspose.from_config(original.get_config())
        replacement.build(x.shape)
        replacement.set_weights(original.get_weights())
        result = np.asarray(replacement(x))
        self.assertEqual(result.shape, expected.shape)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_rejects_kernel_smaller_than_stride(self):
        with self.assertRaises(ValueError):
            SamePaddingConv2DTranspose(4, 1, strides=2)


class TestMakeBackendConsistent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = random_unet()
        cls.x = np.random.default_rng(2).random((2, 256, 256, 3)).astype("float32")
        cls.expected = cls.model.predict(cls.x, verbose=0)

    def test_replaces_all_transposed_convolutions(self):
        converted = make_backend_consistent(self.model, force=True)
        replaced = [layer for layer in converted.layers
                    if isinstance(layer, SamePaddingConv2DTranspose)]
        self.assertEqual(len(replaced), 4)

    def test_forced_conversion_matches_original(self):
        converted = make_backend_consistent(self.model, force=True)
        result = converted.predict(self.x, verbose=0)
        scale = np.abs(self.expected).max()
        self.assertLess(np.abs(result - self.expected).max() / scale, 1e-4)
        np.testing.assert_array_equal(result.argmax(-1), self.expected.argmax(-1))

    def test_unchanged_on_tensorflow(self):
        if keras.backend.backend() != "tensorflow":
            self.skipTest("Only applies to the TensorFlow backend.")
        self.assertIs(make_backend_consistent(self.model), self.model)

    def test_load_unet_and_save_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "unet.keras")
            self.model.save(path)
            loaded = sef.load_unet(path, force=True)
            np.testing.assert_array_equal(
                loaded.predict(self.x, verbose=0).argmax(-1), self.expected.argmax(-1))
            converted_path = os.path.join(directory, "converted.keras")
            loaded.save(converted_path)
            reloaded = keras.saving.load_model(converted_path, compile=False)
            np.testing.assert_allclose(
                reloaded.predict(self.x, verbose=0), loaded.predict(self.x, verbose=0),
                rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
