from models.model import VanillaEncoderDecoder
from models.model import ResidualEncoderDecoder
from models.abstract import ModelRegister
import config
import tensorflow as tf

configuration = config.Config().models

def load_model(name):
  model_family = ModelRegister.instances[name]
  types = configuration[name].architecture.keys()
  combined_model = tf.keras.Sequential(
      [model_family(type_) for type_ in types])
  return combined_model
