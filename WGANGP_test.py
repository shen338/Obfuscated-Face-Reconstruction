import tensorflow as tf
import numpy as np
from WGAN import WGAN
import time
import cv2
import glob


BATCH_SIZE = 20
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
filewriter_path = "./WGAN_v1/filewriter"
checkpoint_path = "./WGAN_v1"
RATIO = 4
BOTTLENECK_NUM = 5
gamma = 0.5

net = WGAN(gamma)
data_path = 'train_espcn.tfrecords'

lr_image = tf.placeholder(tf.float32, shape=[None, 16, 16, 3])

sr_image = net.generate(lr_image)

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

images = tf.image.resize_images(images, (16, 16))
labels = tf.image.resize_images(labels, (64, 64))

image_path = './GANHR_sample/*.png'
address_origin = glob.glob(image_path)

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
        img_lr_input = (img_lr.astype(np.float32) - 127.5) / 127.5

        # img_lr = (np.reshape(cv2.resize(cv2.imread(address_origin[jj]), (16, 16)), [-1, 16, 16, 3]).astype(
        #     np.float32) - 127.5) / 127.5
        # img_lr_2 = (np.reshape(cv2.resize(cv2.imread(address_origin[jj]), (16, 16)), [-1, 16, 16, 3]).astype(
        #     np.float32) - 127.5) / 127.5
        # img_lr = np.stack((img_lr, img_lr_2), 1)
        # img_lr = np.squeeze(img_lr, axis=0)

        img_sr = sess.run([sr_image], feed_dict={lr_image: img_lr})

        img_lr = (img_lr[0, :, :, :]).astype(np.uint8)
        print(img_sr[0].shape)
        print(img_lr.shape)

        img_display = img_sr[0][0, :, :, :]
        print(img_display[0:3, 0:3, 0:2], 'test')
        print(img_hr[1:3, 1:3, 1:3], 'test1')
        img_hr = img_hr[0, :, :, :]
        print(img_hr[0:3, 0:3, 0:2], 'test1')

        img_lr_near = cv2.resize(img_lr, (64, 64), interpolation=cv2.INTER_NEAREST)
        img_lr_cubic = cv2.resize(img_lr, (64, 64), interpolation=cv2.INTER_CUBIC)

        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        img_lr_near = cv2.cvtColor(img_lr_near, cv2.COLOR_BGR2RGB)
        img_lr_cubic = cv2.cvtColor(img_lr_cubic, cv2.COLOR_BGR2RGB)
        cv2.imwrite('C:/Users/JieqiongZhao/PycharmProjects/alexNet_HW6/GANHR_sample/image_' + str(jj) + '_SR' + '.png',
                    img_display)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('C:/Users/JieqiongZhao/PycharmProjects/alexNet_HW6/result_wgangp/image_' + str(jj) + '_LR' + '.png',
        #             img_lr_near)
        # cv2.imwrite('C:/Users/JieqiongZhao/PycharmProjects/alexNet_HW6/result_wgangp/image_' + str(jj) + '_Cubic' + '.png',
        #             img_lr_cubic)
        # cv2.imwrite('C:/Users/JieqiongZhao/PycharmProjects/alexNet_HW6/result_wgangp/image_' + str(jj) + '_HR' + '.png',
        #             img_hr)
        # cv2.imwrite('C:/Users/JieqiongZhao/PycharmProjects/alexNet_HW6/result_wgangp/image_' + str(jj) + '_SR' + '.png',
        #             img_display)
        both = np.vstack((img_lr_near.astype(np.uint8), (img_lr_cubic).astype(np.uint8), img_display.astype(np.uint8), img_hr.astype(np.uint8)))
        # cv2.imwrite('C:/Users/JieqiongZhao/PycharmProjects/alexNet_HW6/result_wgangp/image_' + str(jj) + '_SR' + '.png',
        #             img_display)
        #both = cv2.cvtColor(both, cv2.COLOR_BGR2RGB)
        cv2.namedWindow("Final", 0)

        cv2.imshow('Final', both)
        cv2.resizeWindow("Final", 100, 400)
        cv2.waitKey(0)

    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()




