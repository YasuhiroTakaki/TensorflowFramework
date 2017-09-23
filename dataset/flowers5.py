"""flowers 5class dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tarfile
import tensorflow as tf

REMOTE_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
LOCAL_DIR = os.path.join("data/flowers5/")
ARCHIVE_NAME = "flower_photos.tgz"
DATA_DIR = "flower_photos/"

IMAGE_SIZE = 224
NUM_CLASSES = 5

def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  class_names = []
  for filename in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, filename)
    if os.path.isdir(path):
      class_names.append(filename)

  photo_filenames = []
  photo_labels = []
  class_idx = 0
  for c in class_names:
    directory = os.path.join(dataset_dir, c)
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)
      photo_labels.append(class_idx)
    class_idx = class_idx + 1

  return photo_filenames, sorted(photo_labels)

def get_params():
  """Return dataset parameters."""
  return {
    "image_size": IMAGE_SIZE,
    "num_classes": NUM_CLASSES,
  }

def prepare():
  """Download the flowers 5class dataset."""
  if not os.path.exists(LOCAL_DIR):
    os.makedirs(LOCAL_DIR)
  if not os.path.exists(LOCAL_DIR + ARCHIVE_NAME):
    print("Downloading...")
    urllib.request.urlretrieve(REMOTE_URL, LOCAL_DIR + ARCHIVE_NAME)
  if not os.path.exists(LOCAL_DIR + DATA_DIR):
    print("Extracting files...")
    tar = tarfile.open(LOCAL_DIR + ARCHIVE_NAME)
    tar.extractall(LOCAL_DIR)
    tar.close()

def read(mode):
  dataset_dir = os.path.join(LOCAL_DIR, DATA_DIR)
  photo_filenames, all_labels = _get_filenames_and_classes(dataset_dir)
  return tf.contrib.data.Dataset.from_tensor_slices((photo_filenames, all_labels))

def parse(mode, image_path, label):
  """Parse input record to features and labels."""

  image = tf.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)

  if mode == tf.estimator.ModeKeys.TRAIN:
    image = tf.image.resize_image_with_crop_or_pad(
      image, IMAGE_SIZE + 4, IMAGE_SIZE + 4)
    image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
  else:
    image = tf.image.resize_image_with_crop_or_pad(
      image, IMAGE_SIZE, IMAGE_SIZE)

  image = tf.image.per_image_standardization(image)

  return {"image": image}, {"label": label}
