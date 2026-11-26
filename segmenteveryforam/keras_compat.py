"""Load the U-Net so that it gives the same results on every Keras backend.

The U-Net upsamples with ``Conv2DTranspose(kernel_size=3, strides=2,
padding="same")``. For this combination the full transposed convolution is one
pixel larger than the "same" output, and TensorFlow and PyTorch crop that extra
pixel from opposite sides. A checkpoint trained with TensorFlow therefore
produces outputs shifted by one pixel at each upsampling step when Keras runs
on PyTorch, which changes the segmentation noticeably.

:func:`load_unet` loads a checkpoint and, on backends other than TensorFlow,
swaps these layers for :class:`SamePaddingConv2DTranspose`, which crops the
full output the way TensorFlow does. On TensorFlow the model is returned
unchanged.
"""

import keras


@keras.saving.register_keras_serializable(package="segmenteveryforam")
class SamePaddingConv2DTranspose(keras.layers.Conv2DTranspose):
    """``Conv2DTranspose`` with TensorFlow's ``padding="same"`` on any backend.

    Computes the full ("valid") transposed convolution, then crops it to
    ``input_size * strides`` starting at ``(kernel_size - strides) // 2``, which
    is where TensorFlow places a "same" transposed convolution.
    """

    def __init__(self, *args, **kwargs):
        kwargs["padding"] = "valid"
        super().__init__(*args, **kwargs)
        if self.data_format != "channels_last":
            raise ValueError("SamePaddingConv2DTranspose supports channels_last only.")
        if any(k < s for k, s in zip(self.kernel_size, self.strides)):
            raise ValueError("SamePaddingConv2DTranspose needs kernel_size >= strides.")

    def call(self, inputs):
        outputs = super().call(inputs)
        (ky, kx), (sy, sx) = self.kernel_size, self.strides
        shape = keras.ops.shape(inputs)
        top, left = (ky - sy) // 2, (kx - sx) // 2
        return outputs[:, top:top + shape[1] * sy, left:left + shape[2] * sx, :]

    def compute_output_shape(self, input_shape):
        batch, height, width, _ = input_shape
        (sy, sx) = self.strides
        return (batch,
                None if height is None else height * sy,
                None if width is None else width * sx,
                self.filters)


def _needs_replacement(layer):
    return (isinstance(layer, keras.layers.Conv2DTranspose)
            and not isinstance(layer, SamePaddingConv2DTranspose)
            and layer.padding == "same"
            and layer.data_format == "channels_last"
            and tuple(layer.dilation_rate) == (1, 1)
            and all(k >= s for k, s in zip(layer.kernel_size, layer.strides)))


def make_backend_consistent(model, force=False):
    """Return ``model`` with "same" transposed convolutions made backend-consistent.

    Parameters
    ----------
    model : keras.Model
        A functional Keras model, such as a loaded U-Net checkpoint.
    force : bool, default False
        Replace the layers even on the TensorFlow backend. Useful only for
        testing that the replacement matches TensorFlow.

    Returns
    -------
    keras.Model
        The original model on TensorFlow (unless ``force``), otherwise a copy
        with the same weights in which each affected ``Conv2DTranspose`` is a
        :class:`SamePaddingConv2DTranspose`.
    """
    if not force and keras.backend.backend() == "tensorflow":
        return model
    if not any(_needs_replacement(layer) for layer in model.layers):
        return model

    def clone(layer):
        config = layer.get_config()
        if _needs_replacement(layer):
            return SamePaddingConv2DTranspose.from_config(config)
        return layer.__class__.from_config(config)

    converted = keras.models.clone_model(model, clone_function=clone)
    converted.set_weights(model.get_weights())
    return converted


def load_unet(path, force=False):
    """Load a U-Net checkpoint that behaves the same on every Keras backend.

    Parameters
    ----------
    path : str or path-like
        Path to a ``.keras`` checkpoint.
    force : bool, default False
        Passed to :func:`make_backend_consistent`.

    Returns
    -------
    keras.Model
        The loaded model, ready for ``predict``.
    """
    model = keras.saving.load_model(path, compile=False)
    return make_backend_consistent(model, force=force)
