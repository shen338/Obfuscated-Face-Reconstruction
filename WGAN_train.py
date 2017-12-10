import tensorflow as tf
import numpy as np
from WGAN import WGAN
import time


BATCH_SIZE = 20
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
l_lr = 0.001
filewriter_path = "./WGAN_v2/filewriter"
checkpoint_path = "./WGAN_v2"
RATIO = 4
gamma = 0.5
gan_ratio = 0.1
critic = 5

def train():

    image_lr = tf.placeholder(dtype=tf.float32, shape=(None, 16, 16, 3), name='lr')
    image_hr = tf.placeholder(dtype=tf.float32, shape=(None, 64, 64, 3), name='hr')

    net = WGAN(gamma)

    gen = net.generator(image_lr, bottleneck_num=2)

    real_score = net.discrimintor(gen)
    fake_score = net.discrimintor(image_hr, reuse=True)

    with tf.name_scope('SR_loss'):

        residual = image_hr - gen
        square = tf.abs(residual)
        SR_loss = tf.reduce_mean(square)

        tf.summary.scalar('SR_loss', SR_loss)
    print('test1')

    with tf.name_scope('gan_loss'):

        D_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)

        G_loss = -tf.reduce_mean(fake_score)

        tf.summary.scalar('G_loss', G_loss)
        tf.summary.scalar('D_loss', D_loss)

        G_overall_loss = gan_ratio*G_loss + SR_loss  # this part might need modification

    print('test2')

    # get variable from G and D
    var_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    var_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

    with tf.name_scope('optim'):

        optim_g = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)\
            .minimize(G_overall_loss, var_list=var_g)
        optim_d = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE) \
            .minimize(-D_loss, var_list=var_d)

    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in var_d]

    print('test3')
    # for gradient, var in var_g:
    #     tf.summary.histogram(var.name + '/gradient', gradient)
    #
    # # Add the variables we train to the summary
    # for var in var_g:
    #     tf.summary.histogram(var.name, var)


    # set up logging for tensorboard
    writer = tf.summary.FileWriter(filewriter_path)
    writer.add_graph(tf.get_default_graph())
    summaries = tf.summary.merge_all()

    # saver for storing/restoring checkpoints of the model
    saver = tf.train.Saver()

    data_path = 'train_espcn.tfrecords'

    feature = {'train/image_small': tf.FixedLenFeature([], tf.string),
               'train/image_origin': tf.FixedLenFeature([], tf.string)}

    # create a list of file names
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=NUM_EPOCHS)

    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(filename_queue)

    features = tf.parse_single_example(tfrecord_serialized, features=feature)

    # Convert the image data from string back to the numbers
    image_blur = tf.decode_raw(features['train/image_small'], tf.uint8)
    image_origin = tf.decode_raw(features['train/image_origin'], tf.uint8)

    image_blur = tf.reshape(image_blur, [32, 32, 3])
    image_origin = tf.reshape(image_origin, [128, 128, 3])

    images, labels = tf.train.shuffle_batch([image_blur, image_origin],
                                            batch_size=BATCH_SIZE, capacity=30,
                                            num_threads=16,
                                            min_after_dequeue=10)

    images = tf.image.resize_images(images, (16, 16))
    labels = tf.image.resize_images(labels, (64, 64))

    print('test4')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session() as sess:

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        steps, start_average, end_average = 0, 0, 0
        start_time = time.clock()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for ii in range(NUM_EPOCHS):

            batch_average = 0
            batch_num = int(np.floor(192794 / BATCH_SIZE / 6.0))

            for jj in range(batch_num):

                g_ops = [optim_g, G_loss, summaries]
                d_ops = [optim_d, D_loss]

                for kk in range(critic):

                    steps += 1
                    img_lr, img_hr = sess.run([images, labels])
                    img_lr = (img_lr.astype(np.float32) - 127.5) / 127.5
                    img_hr = (img_hr.astype(np.float32) - 127.5) / 127.5

                    _, loss_d = sess.run(d_ops, feed_dict=
                                  {image_lr: img_lr, image_hr: img_hr})

                    sess.run(clip_D)

                steps += 1
                img_lr, img_hr = sess.run([images, labels])
                img_lr = (img_lr.astype(np.float32) - 127.5) / 127.5
                img_hr = (img_hr.astype(np.float32) - 127.5) / 127.5

                _, loss_g, summary = sess.run(g_ops,
                                feed_dict={image_lr: img_lr, image_hr: img_hr})

                # update W_loss and Kt

                writer.add_summary(summary, steps)
                batch_average += loss_g

                if (steps % 100 == 0):
                    print('step: {:d}, G_loss: {:.9f}, D_loss: {:.9f}'.format(steps, loss_g, loss_d))
                    print('time:', time.clock())

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