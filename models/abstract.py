import tensorflow as tf
import re

class ModelRegister(type):
  instances = {}

  @staticmethod
  def snake_case(string):
    string= re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()
  
  def __init__(cls, *args, **kwargs):
    class_name = ModelRegister.snake_case(cls.__name__).split("_")[0]
    if class_name not in ModelRegister.instances and \
      class_name != "model":
      ModelRegister.instances[class_name] = cls

class Layer(tf.keras.layers.Layer, metaclass=ModelRegister):
  def __init__(self, *args, **kwargs):
    super(Layer, self).__init__(*args, **kwargs)
