"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.client import session as tf_session
from model.slim.mobilenet_v1 import mobilenet_v1
from model.slim.mobilenet_v1 import mobilenet_v1_arg_scope

from common import metrics

slim = tf.contrib.slim

def _get_restore_fn(params):
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if len(params.checkpoint_path) == 0:
    return None

  exclusions = []
  if params.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in params.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(params.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(params.checkpoint_path)
  else:
    checkpoint_path = params.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=params.ignore_missing_vars)

def get_params():
  return {
    "drop_rate": 0.5,
    "weight_decay": 0.00004,
    "checkpoint_path": "",
    "checkpoint_exclude_scopes" : "",
    "ignore_missing_vars" : "",
  }

def model(features, labels, mode, params):
  """CNN classifier model."""
  images = features["image"]
  labels = labels["label"]

  is_training = mode == tf.estimator.ModeKeys.TRAIN

  arg_scope = mobilenet_v1_arg_scope(weight_decay=params.weight_decay)
  with slim.arg_scope(arg_scope):
    logits, end_points = mobilenet_v1(images, params.num_classes, is_training=is_training, prediction_fn=None)

  predictions = tf.argmax(logits, axis=-1)
  loss = tf.losses.sparse_softmax_cross_entropy(
    labels=labels, logits=logits)

  restore_fn = _get_restore_fn(params)
  if not restore_fn == None:
    with tf_session.Session() as session:
      restore_fn(session)

  tf.summary.image("images", images)
#  summary.labeled_image("images", images, predictions)

  metrics = {
    "accuracy": tf.metrics.accuracy(labels, predictions)
  }

  return {"predictions": predictions}, loss, metrics

