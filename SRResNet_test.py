import tensorflow as tf
import numpy as np
from ResNet import ResNet
import time
import cv2
from PSNR import *


BATCH_SIZE = 20
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
filewriter_path = "./SRResNet/filewriter"
checkpoint_path = "./SRResNet"
RATIO = 4
BOTTLENECK_NUM = 5

net = ResNet(LEARNING_RATE, 5, BATCH_SIZE, 4)
data_path = 'train_espcn.tfrecords'

lr_image = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])


sr_image = net.generate(lr_image, BOTTLENECK_NUM)

# saver for storing/restoring checkpoints of the model
saver = tf.train.Saver()

feature = {'train/image_small': tf.FixedLenFeature([], tf.string),
           'train/image_origin': tf.FixedLenFeature([], tf.string)}

# create a list of file names
filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

reader = tf.TFRecordReader()
_, tfrecord_serialized = reader.read(filename_queue)

features = tf.parse_single_example(tfrecord_serialized, features=feature)

# Convert the image data from string back to the numbers
image_small = tf.decode_raw(features['train/image_small'], tf.uint8)
image_origin = tf.decode_raw(features['train/image_origin'], tf.uint8)

image_small = tf.reshape(image_small, [32, 32, 3])
image_origin = tf.reshape(image_origin, [128, 128, 3])

step = 1

images, labels = tf.train.shuffle_batch([image_small, image_origin],
                                        batch_size=4, capacity=30,
                                        num_threads=1,
                                        min_after_dequeue=10)

# images = tf.image.resize_images(images, (16,16))
# images = tf.image.resize_bilinear(images, (32,32))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    net.load(sess, saver, checkpoint_path)

    steps, start_average, end_average = 0, 0, 0
    start_time = time.clock()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for jj in range(200):
        steps += 1
        img_lr, img_hr = sess.run([images, labels])
        img_lr = img_lr.astype(np.float32) / 255


        img_sr = sess.run([sr_image], feed_dict={lr_image: img_lr})

        img_lr = (img_lr[0, :, :, :] * 255).astype(np.uint8)
        print(img_sr[0].shape)
        print(img_lr.shape)

        img_display = img_sr[0][0, :, :, :]
        img_hr = img_hr[0, :, :, :]
        img_lr_near = cv2.resize(img_lr, (128, 128), interpolation=cv2.INTER_NEAREST)
        img_lr_cubic = cv2.resize(img_lr, (128, 128), interpolation=cv2.INTER_CUBIC)

        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        img_lr_near = cv2.cvtColor(img_lr_near, cv2.COLOR_BGR2RGB)
        img_lr_cubic = cv2.cvtColor(img_lr_cubic, cv2.COLOR_BGR2RGB)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('C:/Users/JieqiongZhao/PycharmProjects/alexNet_HW6/result/image_' + str(jj) + '_LR' + '.png',
        #             img_lr_near)
        # cv2.imwrite('C:/Users/JieqiongZhao/PycharmProjects/alexNet_HW6/result/image_' + str(jj) + '_Cubic' + '.png',
        #             img_lr_cubic)
        # cv2.imwrite('C:/Users/JieqiongZhao/PycharmProjects/alexNet_HW6/result/image_' + str(jj) + '_HR' + '.png',
        #             img_hr)
        # cv2.imwrite('C:/Users/JieqiongZhao/PycharmProjects/alexNet_HW6/result/image_' + str(jj) + '_SR' + '.png',
        #             img_display)
        #both = cv2.cvtColor(both, cv2.COLOR_BGR2RGB)
        print('PSNR', psnr(img_hr, img_display))
        cv2.namedWindow("Final", 0)
        both = np.hstack((img_lr_near, img_display, img_hr))
        cv2.imshow('Final', both)
        cv2.resizeWindow("Final", 1000, 500)
        cv2.waitKey(0)

    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()




