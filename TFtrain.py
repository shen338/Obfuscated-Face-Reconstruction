
import tensorflow as tf
import numpy as np
import cv2
import glob
import random
import sys


def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    #img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.uint8)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


shuffle_data = True
image_path_small = './origin/small/*.png'
address_small = glob.glob(image_path_small)
print(len(address_small))
image_path_origin = './origin/origin/*.png'
address_origin = glob.glob(image_path_origin)

if shuffle_data:
    c = list(zip(address_small, address_origin))
    random.shuffle(c)
    address_small,  address_origin= zip(*c)

train_filename = 'train_espcn.tfrecords'

# create new TFrecord file
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(address_small)):

    if not i % 1000:
        print('Train data: {}/{}'.format(i, len(address_small)))
        sys.stdout.flush()

    img_small = load_image(address_small[i])
    img_origin = load_image(address_origin[i])

    feature = {'train/image_small': _bytes_feature(tf.compat.as_bytes(img_small.tostring())),
               'train/image_origin': _bytes_feature(tf.compat.as_bytes(img_origin.tostring()))}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()
