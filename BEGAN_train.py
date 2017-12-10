import tensorflow as tf
import numpy as np
from BEGAN import BEGAN
import time


BATCH_SIZE = 20
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
l_lr = 0.001
filewriter_path = "./BEGAN_v1/filewriter"
checkpoint_path = "./BEGAN_v1"
RATIO = 4
gamma = 0.5
gan_ratio = 0.1

def train():

    image_lr = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='lr')
    image_hr = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 3), name='hr')
    lamb = tf.placeholder(dtype=tf.float32, name='Kt')
    net = BEGAN(gamma)

    gen = net.generator(image_lr)

    dis_fake = net.discrimintor(gen)
    dis_real = net.discrimintor(image_hr, reuse=True)

    with tf.name_scope('SR_loss'):

        residual = image_hr - gen
        square = tf.abs(residual)
        SR_loss = tf.reduce_mean(square)

        tf.summary.scalar('SR_loss', SR_loss)

    print('test1')
    with tf.name_scope('gan_loss'):

        D_real = net.loss(image_hr, dis_real, type='L1')
        D_fake = net.loss(gen, dis_fake, type='L1')
        D_loss = D_real - lamb*D_fake

        G_gan_loss = D_fake

        W_loss = D_real + tf.abs(gamma*D_real - G_gan_loss)

        G_loss = gan_ratio*G_gan_loss + SR_loss

        tf.summary.scalar('Kt', lamb)
        tf.summary.scalar('G_gan_loss', G_gan_loss)
        tf.summary.scalar('G_loss', G_loss)
        tf.summary.scalar('D_loss', D_loss)
        tf.summary.scalar('W_loss', W_loss)

    print('test2')

    # get variable from G and D
    var_g = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'generator')
    var_d = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'discriminator')


    with tf.name_scope('optim'):

        optim_g = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)\
            .minimize(G_loss, var_list=var_g)
        optim_d = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE) \
            .minimize(D_loss, var_list=var_d)
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
    print('test4')
    Kt = 0

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
            batch_num = int(np.floor(192794 / BATCH_SIZE))

            for jj in range(batch_num):

                steps += 1
                img_lr, img_hr = sess.run([images, labels])
                img_lr = img_lr.astype(np.float32) / 255
                img_hr = img_hr.astype(np.float32) / 255

                g_ops = [optim_g, G_loss, D_real, D_fake]
                d_ops = [optim_d, D_loss, summaries]

                _, loss_g, loss_real, loss_fake = sess.run(g_ops,
                        feed_dict={image_lr: img_lr, image_hr: img_hr, lamb: Kt})

                _, loss_d, summary = sess.run(d_ops, feed_dict=
                                  {image_lr: img_lr, image_hr: img_hr, lamb: Kt})

                # update W_loss and Kt
                Kt = np.maximum(np.minimum(1., Kt + l_lr * (gamma * loss_real - loss_fake)), 0.)
                global_loss = loss_real + np.abs(gamma * loss_real - loss_fake)
                writer.add_summary(summary, steps)
                batch_average += global_loss

                if (jj % 100 == 0):
                    print('step: {:d}, loss: {:.9f}'.format(steps, global_loss))
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