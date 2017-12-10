import tensorflow as tf
import numpy as np
from ResNet import *
import time
from VGG import *

BATCH_SIZE = 20
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
filewriter_path = "./SRResNet/filewriter"
checkpoint_path = "./SRResNet"
RATIO = 4
BOTTLENECK_NUM = 5

def train():

    image_lr = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3))
    image_hr = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3))

    net = ResNet(LEARNING_RATE, 5, BATCH_SIZE, 4)

    output = net.generator(image_lr, BOTTLENECK_NUM)

    #para_dict = load_model('vgg16.npy')

    print("test1")

    with tf.name_scope('content_loss'):

        loss = net.abs_loss(output, image_hr)

    print("test2")

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        print('test')
        trainable = tf.trainable_variables()
        optim = optimizer.minimize(loss, var_list=trainable)
    print("test")
    # Add the variables we train to the summary
    for var in trainable:
        tf.summary.histogram(var.name, var)
    print("test3")
    # set up logging for tensorboard
    writer = tf.summary.FileWriter(filewriter_path)
    writer.add_graph(tf.get_default_graph())
    summaries = tf.summary.merge_all()

    # saver for storing/restoring checkpoints of the model
    saver = tf.train.Saver()

    data_path = 'train_espcn.tfrecords'
    print("test4")
    feature = {'train/image_small': tf.FixedLenFeature([], tf.string),
               'train/image_origin': tf.FixedLenFeature([], tf.string)}


    # create a list of file names
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=NUM_EPOCHS)
    print("test5")
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(filename_queue)

    features = tf.parse_single_example(tfrecord_serialized, features=feature)

    # Convert the image data from string back to the numbers
    image_small = tf.decode_raw(features['train/image_small'], tf.uint8)
    image_origin = tf.decode_raw(features['train/image_origin'], tf.uint8)
    print("test6")
    image_small = tf.reshape(image_small, [32, 32, 3])
    image_origin = tf.reshape(image_origin, [128, 128, 3])



    images, labels = tf.train.shuffle_batch([image_small, image_origin],
                                            batch_size=BATCH_SIZE, capacity=30,
                                            num_threads=16,
                                            min_after_dequeue=10)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        steps, start_average, end_average = 0, 0, 0
        start_time = time.clock()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for ii in range(NUM_EPOCHS):

            batch_average = 0
            batch_num = int(np.floor(192794/BATCH_SIZE))

            for jj in range(batch_num):

                steps = 1
                img_lr, img_hr = sess.run([images, labels])
                img_lr = img_lr.astype(np.float32)/255
                img_hr = img_hr.astype(np.float32)/255
                #print(img_hr.shape)

                summary, loss_value, _ = sess.run([summaries, loss, optim],
                                                  feed_dict={image_lr: img_lr,
                                                             image_hr: img_hr})
                if (jj % 100 == 0):
                    print('step: {:d}, loss: {:.9f}'.format(steps, loss_value))
                    print('time:' , time.clock())
                writer.add_summary(summary, steps)
                batch_average += loss_value


            batch_average = float(batch_average) / batch_num

            duration = time.time() - start_time
            print('Epoch: {}, step: {:d}, loss: {:.9f}, '
                  '({:.3f} sec/epoch)'.format(ii, steps, batch_average, duration))

            start_time = time.time()
            net.save(sess, saver, checkpoint_path, steps)

        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)
        sess.close()



if __name__ == '__main__':
    train()