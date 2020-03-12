import pickle
import tensorflow as tf
import config
import pandas as pd

configuration = config.Config()

def preprocess_data(x, y, mean, standard_deviation):
 x_normalized = ((x - mean) / standard_deviation)
 y_normalized = ((y - mean) / standard_deviation)
 return x_normalized, y_normalized

def load_dataset(path, key="training"):
  strategy = tf.distribute.get_strategy()
  dataset, length, column_names = load_pickle_dataset(path)
  configuration.dataset.columns = column_names 
  if configuration.dataset.shuffle:
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
  dataset = dataset.batch(configuration.dataset[key].batch_size)
  if configuration.dataset[key].repeat:
    if configuration.dataset[key].repeat < 0:
      dataset = dataset.repeat()
    else:
      dataset = dataset.repeat(configuration.dataset[key].repeat)

  return iter(strategy.experimental_distribute_dataset(dataset))

def load_pickle_dataset(pickle_path, cache=True):  
  with tf.io.gfile.GFile(pickle_path, "rb") as f:
    dataframe = pickle.load(f)
  target = source = tf.cast(dataframe.values, tf.float32)
  standard_deviation = tf.math.reduce_std(source, axis=0)
  mean = tf.reduce_mean(source, axis=0)
  dataset = tf.data.Dataset.from_tensor_slices((source, target))
  dataset = dataset.map(lambda x, y: \
              preprocess_data(x, y, mean, standard_deviation),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if cache:
    dataset = dataset.cache(configuration.dataset.records_path)
  return (dataset,
          len(dataframe),
          dataframe.columns.values.tolist())
