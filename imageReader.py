"""
Tiny ImageNet: Input Pipeline
Written by Patrick Coady (pcoady@alum.mit.edu)

Reads in jpegs, distorts images (flips, translations, hue and
saturation) and builds QueueRunners to keep the GPU well-fed. Uses
specific directory and file naming structure from data download
link below.

Also builds dictionary between label integer and human-readable
class names.

Get data here:
https://tiny-imagenet.herokuapp.com/
"""
import glob
import re
import tensorflow as tf
import random
import numpy as np


from queue import Queue


def load_filenames_labels(mode):
  """Gets filenames and labels

  Args:
    mode: 'train' or 'val'
      (Directory structure and file naming different for
      train and val datasets)

  Returns:
    list of tuples: (jpeg filename with path, label)
  """
  label_dict, class_description = build_label_dicts()
  filenames_labels = []
  if mode == 'train':
    filenames = glob.glob('tiny-imagenet-200/train/*/images/*.JPEG')

    for filename in filenames:
      match = re.search(r'n\d+', filename)
      label = str(label_dict[match.group()])
      filenames_labels.append((filename, label))
      #print(filename)
  elif mode == 'val':
    with open('tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
      for line in f.readlines():
        split_line = line.split('\t')
        filename = 'tiny-imagenet-200/val/images/' + split_line[0]
        label = str(label_dict[split_line[1]])
        filenames_labels.append((filename, label))

  return filenames_labels


def build_label_dicts():
  """Build look-up dictionaries for class label, and class description

  Class labels are 0 to 199 in the same order as
    tiny-imagenet-200/wnids.txt. Class text descriptions are from
    tiny-imagenet-200/words.txt

  Returns:
    tuple of dicts
      label_dict:
        keys = synset (e.g. "n01944390")
        values = class integer {0 .. 199}
      class_desc:
        keys = class integer {0 .. 199}
        values = text description from words.txt
  """
  label_dict, class_description = {}, {}
  with open('tiny-imagenet-200/wnids.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset = line[:-1]  # remove \n
      label_dict[synset] = i
  with open('tiny-imagenet-200/words.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset, desc = line.split('\t')
      desc = desc[:-1]  # remove \n
      if synset in label_dict:
        class_description[label_dict[synset]] = desc
  return label_dict, class_description


def read_image(filename_q, mode):
  """
  Load next jpeg file from filename / label queue
  Randomly applies distortions if mode == 'train' (including a
  random crop to [56, 56, 3]). Standardizes all images.

  Args:
    filename_q: Queue with 2 columns: filename string and label string.
     filename string is relative path to jpeg file. label string is text-
     formatted integer between '0' and '199'
    mode: 'train' or 'val'

  Returns:
    [img, label]:
      img = tf.uint8 tensor [height, width, channels]  (see tf.image.decode.jpeg())
      label = tf.unit8 target class label: {0 .. 199}

  """
  item = filename_q.get()
  filename = item[0]
  label = item[1]
  file = tf.read_file(filename)
  img = tf.image.decode_jpeg(file, channels=3)
  img = tf.image.resize_images(img, np.array([227,227]))
  # image distortions: left/right, random hue, random color saturation
  # if mode == 'train':
  #   img = tf.random_crop(img, np.array([227, 227, 3]))
  #   img = tf.image.random_flip_left_right(img)
  #   # val accuracy improved without random hue
  #   # img = tf.image.random_hue(img, 0.05)
  #   img = tf.image.random_saturation(img, 0.5, 2.0)
  # else:
  #   img = tf.image.crop_to_bounding_box(img, 4, 4, 227, 227)

  label = tf.string_to_number(label, tf.int32)
  label = tf.cast(label, tf.uint8)
  return img, label


def single_batch(filenames_queue, mode, config):
    """
    :param filenames_queue: A Queue with 2 columns: filename string and label string.
    :param mode: 'train' or 'val'
    :param config:  training configuration object
    :return: imgs: tf.uint8 tensor [batch_size, height, width, channels]
                   labels: tf.uint8 tensor [batch_size,num_class](one-hot)
    """
    batch_size = config.batch_size
    imgs = []
    labels = []
    for i in range(batch_size):
         items = read_image(filenames_queue, mode)
         imgs.append(items[0])
         labels.append(items[1])
    imgs = tf.stack(imgs)
    labels = tf.stack(labels)

    # onehot encoding for labels
    labels = tf.one_hot(labels, depth=200)
    return [imgs, labels]

# class TrainConfig(object):
#   """Training configuration"""
#   batch_size = 50
#   num_epochs = 50
#   summary_interval = 250
#   eval_interval = 2000  # must be integer multiple of summary_interval
#   lr = 0.001  # learning rate
#   reg = 5e-4  # regularization
#   momentum = 0.9
#   dropout_keep_prob = 0.5
#   #model_name = 'AlexNet'  # choose model
#   #model = staticmethod(globals()[model_name])  # gets model by name
#
# config = TrainConfig()
# mode = 'val'
# filenames_labels = load_filenames_labels(mode)
# print(filenames_labels[1:10])
# random.shuffle(filenames_labels)
# print(filenames_labels[0][0])
# q = Queue()
# for items in filenames_labels:
#     q.put(items)
# print(q.qsize())
# batch = single_batch(q,mode,config)
#
# print(batch[0].shape)
#
# with tf.Session() as sess:
#     darray = batch[0].eval().astype(np.uint8)
# print(darray[3,:,:,:].reshape((227,227,3)))
# plt.imshow(darray[3,:,:,:].reshape((227,227,3)))
# plt.show()

