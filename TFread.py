import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import time
import cv2
data_path = 'train_espcn.tfrecords'

feature = {'train/image_small': tf.FixedLenFeature([], tf.string),
           'train/image_origin': tf.FixedLenFeature([], tf.string)}

# create a list of file names
filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

print(filename_queue)
reader = tf.TFRecordReader()
_, tfrecord_serialized = reader.read(filename_queue)

features = tf.parse_single_example(tfrecord_serialized, features=feature)

# Convert the image data from string back to the numbers
image_small = tf.decode_raw(features['train/image_small'], tf.uint8)
image_origin = tf.decode_raw(features['train/image_origin'], tf.uint8)

image_small = tf.reshape(image_small, [32, 32, 3])
image_origin = tf.reshape(image_origin, [128, 128, 3])

images, labels = tf.train.shuffle_batch([image_small, image_origin], batch_size=50, capacity=30, num_threads=16,
                                            min_after_dequeue=10)

with tf.Session() as sess:

    print(images)
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    print(images)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for batch_index in range(10):
        print(time.clock())

        img, lbl = sess.run([images, labels])
        print(time.clock())
        img = img.astype(np.uint8)
        print(img.shape)
        # #lbl = lbl.astype(np.uint8)
        # for j in range(10):
        #     plt.subplot(2, 5, j + 1)
        #     plt.imshow(lbl[j, ...])
        #     plt.title('ss')
        # plt.show()

    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()

