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
  "root_mean_squared_error": losses.rmse_loss_fn,
  "diff_loss": losses.diff_loss_fn
}

def get_learning_rate(step):
  if configuration.training.learning_rate.step:
    if step >= configuration.training.learning_rate.step[0]:
      configuration.training.learning_rate.step.pop(0)
      return configuration.training.learning_rate.value.pop(0)

def plot_diff_loss(diff_loss):
  for idx, column_name in \
      enumerate(configuration.dataset.get("columns", list())):
    scalar = diff_loss[idx]
    tf.summary.scalar(column_name, scalar,
                      tf.summary.experimental.get_step())

def save_checkpoint(checkpoint):
  checkpoint.save(file_prefix=configuration.training.checkpoint_folder)

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
