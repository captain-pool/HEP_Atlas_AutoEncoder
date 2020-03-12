import itertools
import tensorflow as tf
import config
from models import abstract

class VanillaEncoderDecoder(abstract.Layer):
  def __init__(self, type_, *args, **kwargs):
    super(VanillaEncoderDecoder, self).__init__(name=type_, *args, **kwargs)
    self._config = config.Config().models
    self._denses = [
        tf.keras.layers.Dense(num_units) \
        for num_units in self._config.vanilla.architecture[type_].num_units
    ]
    self._activation = tf.nn.swish
    self._last_activation = tf.nn.tanh if type_.lower() == "decoder" \
                            else lambda x: x

  def call(self, inputs):
    intermediate = inputs
    # Layer Stitching
    for dense in self._denses[:-1]:
      intermediate = dense(intermediate)
      intermediate = self._activation(intermediate)
    intermediate = self._last_activation(self._denses[-1](intermediate))
    return intermediate

class ResidualEncoderDecoder(abstract.Layer):
  def __init__(self, type_, *args, **kwargs):
    super(ResidualEncoderDecoder, self).__init__(name=type_, *args, **kwargs)
    self._config = config.Config().models
    self._weight = self._config.residual.weight
    num_denses = self._config.residual.architecture[type_].num_units
    self._dense_units = itertools.chain(*zip(num_denses, num_denses))
    self._denses = iter([
        tf.keras.layers.Dense(num_units) \
        for num_units in self._dense_units ])
    self._activation = tf.keras.activations.tanh

  def call(self, inputs):
    initial = stitch = intermediate = self._activation(
        next(self._denses)(inputs))
    for idx, dense in enumerate(self._denses):
      intermediate = dense(intermediate)
      if not idx % 2:
        breakpoint()
        intermediate = stitch = stitch + self._weight * intermediate
    return initial + self._weight * stitch
