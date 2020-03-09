import functools
import tensorflow as tf
import config
import losses

configuration = config.Config() #pylint: disable=no-value-for-parameter

_STRATEGIES = {
    "cpu": [
        functools.partial(tf.distribute.OneDeviceStrategy, "/cpu:0"),
       "/cpu:0"],
    "gpu": [
        functools.partial(tf.distribute.OneDeviceStrategy, "/gpu:0"),
        "/gpu:0"]
}

_LOSSES = {
  "root_mean_squared_error": losses.rmse_loss_fn
}

def save_checkpoint(checkpoint):
  checkpoint.save(configuration.training.checkpoint_folder)

def checkpoint_exists():
  checkpoint_prefix = configuration.training.checkpoint_folder
  return tf.io.gfile.exists(checkpoint_prefix)\
         and tf.train.latest_checkpoint(checkpoint_prefix)

def load_latest_checkpoint(checkpoint):
  return checkpoint.restore(
      tf.train.latest_checkpoint(
          configuration.training.checkpoint_folder))

def get_loss(loss_name=None):
  loss_name = loss_name or configuration.training.loss
  loss_fn = _LOSSES.get(loss_name)

  if not loss_fn:
    loss_fn = tf.keras.losses.get(loss_name)

  return loss_fn

def get_strategy(device):
  if "gpu" in device:
    assert tf.test.is_gpu_available(), "No GPUs Found."
  strategy, device = _STRATEGIES[device]
  strategy = strategy()
  return strategy, device
