from models.model import VanillaEncoderDecoder
from models.model import ResidualEncoderDecoder
from models.abstract import ModelRegister
import config
import tensorflow as tf

configuration = config.Config().models

class EncoderDecoder(tf.keras.Model):
  def __init__(self, types, model_family, *args, **kwargs):
    super(EncoderDecoder, self).__init__(*args, **kwargs)
    self._types = types
    for type_ in self._types:
      setattr(self, type_, model_family(type_))

  def call(self, inputs):
    x = inputs
    for type_ in self._types:
      x = vars(self)[type_](x)
    return x

def load_model(name):
  model_family = ModelRegister.instances[name]
  types = configuration[name].architecture.keys()
  #combined_model = tf.keras.models.Sequential([model_family(type_) for type_ in types])
  combined_model = EncoderDecoder(types, model_family, name="combined_model") 
  return combined_model
