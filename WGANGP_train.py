import tensorflow as tf
import numpy as np
from WGAN import WGAN
import time


BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0002
l_lr = 0.001
filewriter_path = "./WGANGP_v3/filewriter"
checkpoint_path = "./WGANGP_v3"
RATIO = 4
gamma = 0.5
gan_ratio = 0.01
critic = 5
gp_rate = 10

def train():

    image_lr = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 16, 16, 3), name='lr')
    image_hr = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 64, 64, 3), name='hr')
    #sigma = tf.placeholder(dtype=tf.float32, name='sigma')
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

        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        gp_sample = interpolate(gen, image_hr)

        # sigma = tf.random_uniform(
        #     shape=[BATCH_SIZE, 1],
        #     minval=0.,
        #     maxval=1.
        # )
        #
        # gp_sample = gen*sigma + image_hr*(1 - sigma)

        #gp_sample = tf.reshape(gp_sample, [-1, 128, 128, 3])

        print(gen.get_shape(),'test2')

        print(image_hr.get_shape())

        print(gp_sample.get_shape())

        gp_gradient = tf.gradients(net.discrimintor(gp_sample, reuse=True), gp_sample)

        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradient[0]), reduction_indices=[-1]))

        gp_loss = tf.reduce_mean(tf.square(grad_norm-1.))

        D_overall_loss = D_loss + gp_rate*gp_loss

        tf.summary.scalar('G_loss', (G_loss))
        tf.summary.scalar('D_loss', (D_loss))
        tf.summary.scalar('GP_loss', gp_loss)

        G_overall_loss = gan_ratio*G_loss + SR_loss  # this part might need modification

    print('test2')

    # get variable from G and D
    var_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
    var_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

    with tf.name_scope('optim'):

        optim_g = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5, beta2=0.9)\
            .minimize(G_overall_loss, var_list=var_g)
        optim_d = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1=0.5, beta2=0.9) \
            .minimize(D_overall_loss, var_list=var_d)

    # gradient penalty


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
    loss_d = 0

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

                g_ops = [optim_g, G_overall_loss]
                d_ops = [optim_d, D_overall_loss]

                for kk in range(critic):

                    steps += 1

                    img_lr, img_hr = sess.run([images, labels])
                    img_lr = (img_lr.astype(np.float32) - 127.5) / 127.5
                    img_hr = (img_hr.astype(np.float32) - 127.5) / 127.5

                    _, loss_d = sess.run(d_ops, feed_dict=
                                  {image_lr: img_lr, image_hr: img_hr})

                steps += 1

                img_lr, img_hr = sess.run([images, labels])
                img_lr = (img_lr.astype(np.float32) - 127.5) / 127.5
                img_hr = (img_hr.astype(np.float32) - 127.5) / 127.5

                _, loss_g = sess.run(g_ops,
                                feed_dict={image_lr: img_lr, image_hr: img_hr})


                if steps%10 == 0:
                    summary = sess.run(summaries, feed_dict={image_lr: img_lr, image_hr: img_hr})
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