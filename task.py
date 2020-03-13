import collections
import importlib
import os

import argparse
import config
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm

def difference_metric(saved_model, *args):
  diff_loss_fn = utils.get_loss("diff_loss")

  @tf.function
  def difference_metric_fn(X, Y):
    Y_hat = saved_model(X)
    loss = diff_loss_fn(Y_hat, Y)
    return tf.squeeze(loss)

  return difference_metric_fn


def predictions(saved_model, mean, std):
  @tf.function
  def prediction_fn(X, Y):
    Y_hat = saved_model(X)
    Y_hat = (Y_hat * std) + mean
    return tf.squeeze(Y_hat)
  return prediction_fn

_METRICS_REGISTER = {
    #"difference": difference_metric,
    "predictions": predictions
}

@tf.function
def summary_histogram_fn(summary_writer, name, step, buckets, value):
  with summary_writer.as_default():
    tf.summary.histogram(name, value, step=step, buckets=buckets)  

def plot_metrics(summary_writer, metrics, export_jpegs, columns, nbins):
    for metric_name in tqdm.tqdm(metrics.keys(), position=0):
      for index, values in tqdm.tqdm(
          enumerate(metrics[metric_name]),
          total=len(metrics[metric_name]), position=1):
        for idx, value in enumerate(values):
          summary_histogram_fn(
            summary_writer,
            "%s_%s" % (columns[idx], metric_name),
                               tf.convert_to_tensor(index, dtype=tf.int64),
                               tf.convert_to_tensor(nbins),
                               tf.convert_to_tensor(value))
      if export_jpegs:
        Ys = np.asarray(metrics[metric_name]).T
        Xs = list(range(len(metrics[metric_name])))
        breakpoint()
        for idx, y in enumerate(Ys):
          plt.title("%s: (%s)" % (metric_name, columns[idx])) 
          plt.hist(y, bins=nbins)
          figure_path = os.path.join(export_jpegs,
                                     "%s_%s.jpg" % (metric_name, columns[idx]))
          plt.savefig(figure_path)
          plt.clf()
          print("Saved plot of %s to %s" % (metric_name, figure_path))


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--saved_model_path", type=os.path.normpath,
      required=True, help="Path to load the Saved Model from")
  parser.add_argument(
      "--nbins", type=int, required=True, help="Number of bins of Histogram")
  parser.add_argument(
      "--export_jpegs", type=os.path.normpath, default=" ",
      help="Path to export th plots as JPEGs")
  parser.add_argument(
      "--configs", dest="config",
      type=os.path.normpath,
      required=True,
      nargs="+",
      help="Path to Config Files")
  parser.add_argument(
      "--infer_pickle_path",
      type=os.path.normpath, default=None,
      help="Path to Pickled Pandas Dataframe (Test)")
  parser.add_argument(
      "--logdir", type=os.path.normpath, required=True, help="Path to logdir")
  return parser

def main(argv):
  saved_model = tf.saved_model.load(argv.saved_model_path)
  returned_values = (dataset
                     .load_pickle_dataset(argv.infer_pickle_path)) 
  infer_dataset, dataset_length, column_names, mean, std = returned_values
  infer_dataset = infer_dataset.batch(1)
  metrics = collections.defaultdict(list)
  metric_fns = dict()
  inference_dir = os.path.join(argv.logdir, "inference")
  if not tf.io.gfile.exists(inference_dir):
    tf.io.gfile.makedirs(inference_dir)
  summary_writer = tf.summary.create_file_writer(inference_dir)
  for metric in _METRICS_REGISTER:
    metric_fns[metric] = _METRICS_REGISTER[metric](saved_model, mean, std)

  for X, Y in tqdm.tqdm(infer_dataset, total=dataset_length):
    for metric in metric_fns:
      value = metric_fns[metric](X, Y).numpy()
      metrics[metric].append(value)
  plot_metrics(summary_writer, metrics,
               argv.export_jpegs.strip(), column_names, argv.nbins)

if __name__ == "__main__":
  parser = build_parser()
  args = parser.parse_args()
  configs = config.Config(args)
  globals().update({
      "dataset": importlib.import_module("dataset"),
      "utils": importlib.import_module("utils")
  })
  main(configs)
