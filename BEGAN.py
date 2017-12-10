import tensorflow as tf
import numpy as np
import sys
import os


class BEGAN:

    def __init__(self, gamma):

        self.gamma = gamma

    def generator(self, input):

        with tf.variable_scope('generator'):
            init = tf.contrib.layers.xavier_initializer()

            x1 = tf.layers.conv2d(input, 64, 4, strides=(2, 2),
                                  padding='same', kernel_initializer=init,
                                  activation=tf.nn.elu, name='downsample1_1')

            # x1 16x16
            x2 = tf.layers.conv2d(x1, 128, 4, strides=(2, 2),
                                  padding='same', kernel_initializer=init,
                                  activation=tf.nn.elu, name='downsample2_1')

            x2 = tf.layers.batch_normalization(x2, name='bn1')
            # x2 8x8

            x3 = tf.layers.conv2d(x2, 256, 4, strides=(2, 2),
                                  padding='same', kernel_initializer=init,
                                  activation=tf.nn.elu, name='downsample3_1')

            x3 = tf.layers.conv2d(x3, 256, 4, strides=(1, 1),
                                  padding='same', kernel_initializer=init,
                                  activation=tf.nn.elu, name='downsample3_2')

            # x3 4x4

            x4 = tf.layers.conv2d_transpose(x3, 128, 4, strides=(2, 2),
                                            padding='same', kernel_initializer=init,
                                            activation=tf.nn.elu, name='upsample1_2')

            x4 = tf.layers.batch_normalization(x4, name='bn2')
            # x4 8x8

            x = x4 + x2
            x5 = tf.layers.conv2d_transpose(x, 64, 4, strides=(2, 2),
                                            padding='same', kernel_initializer=init,
                                            activation=tf.nn.elu, name='upsample2_1')
            x5 = tf.layers.batch_normalization(x5, name='bn3')
            # x5 16x16

            x = x5 + x1
            x = tf.layers.conv2d_transpose(x, 48, 4, strides=(2, 2),
                                           padding='same', kernel_initializer=init,
                                           activation=tf.nn.elu, name='upsample3_1')

            # pixel shuffle
            output = self.PS(x, 4, color=True)

            return output

    def discrimintor(self, input, reuse=False):
        with tf.variable_scope('discriminator') as scope:

            if reuse:
                scope.reuse_variables()
            init = tf.contrib.layers.xavier_initializer()

            # 64x64
            x = tf.layers.conv2d(input, 64, 4, strides=(2, 2),
                                 padding='same', kernel_initializer=init,
                                 activation=tf.nn.elu, name='D_downsample_1')

            # 32x32
            x = tf.layers.conv2d(x, 128, 4, strides=(2, 2),
                                 padding='same', kernel_initializer=init,
                                 activation=tf.nn.elu, name='D_downsample_2')

            # 16x16
            x = tf.layers.conv2d(x, 256, 4, strides=(2, 2),
                                 padding='same', kernel_initializer=init,
                                 activation=tf.nn.elu, name='D_downsample_3')

            # 32x32
            x = self.tensor_resize(x, 32, name='resize_1')

            x = tf.layers.conv2d(x, 128, 4, strides=(1, 1),
                                 padding='same', kernel_initializer=init,
                                 activation=tf.nn.elu, name='D_upsample_1')

            # 64x64
            x = self.tensor_resize(x, 64, name='resize_2')

            x = tf.layers.conv2d(x, 64, 4, strides=(1, 1),
                                 padding='same', kernel_initializer=init,
                                 activation=tf.nn.elu, name='D_upsample_2')

            # 128x128
            x = self.tensor_resize(x, 128, name='resize_3')

            x = tf.layers.conv2d(x, 3, 4, strides=(1, 1),
                                 padding='same', kernel_initializer=init,
                                 activation=tf.nn.elu, name='D_upsample_3')

            return x

    def tensor_resize(self, tensor, dim, name=None):

        return tf.image.resize_nearest_neighbor(tensor,
                                                size=(int(dim), int(dim)), name=name)

    def gan_loss(self, hr_image, D_hr_image, G_image, D_G_image, Kt):

        G_loss = self.loss(hr_image, G_image, type='L1')

        D_real = self.loss(hr_image, D_hr_image, type='L1')
        D_fake = self.loss(G_image, D_G_image, type='L1')
        D_loss = D_real - Kt*D_fake

        W_loss = D_real + tf.abs(self.gamma*D_real - G_loss)

        return G_loss, D_loss, W_loss

    def leaky_relu(self, x, alpha=0.01):
        """
        Compute the leaky ReLU activation function.

        """

        # If x is below 0 returns alpha*x else it will return x.
        activation = tf.maximum(x, alpha * x)
        return activation

    def loss(self, output, target, type='L1'):
        if type=='L2':
            residual = output - target
            square = tf.square(residual)
            reduce_loss = tf.reduce_mean(square)

        else:
            residual = output - target
            square = tf.abs(residual)
            reduce_loss = tf.reduce_mean(square)

        return reduce_loss

    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        print(I.get_shape())
        X = tf.reshape(I, (-1, a, b, r, r))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (-1, a * r, b * r, 1))

    def PS(self, X, r, color=False, name=None):
        # Main OP that you can arbitrarily use in you tensorflow code
        with tf.name_scope(name):
            if color:
                Xc = tf.split(X, 3, axis=3)
                X = tf.concat([self._phase_shift(x, r) for x in Xc], 3)
            else:
                X = self._phase_shift(X, r)
            return X

    def save(self, sess, saver, logdir, step):
        '''
        :param sess:  tf.Session()
        :param saver:  tf.summary.FileWriter
        :param logdir:  log directory for the store the model
        :param step:   training step number
        :return:
        '''
        sys.stdout.flush()

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        checkpoint = os.path.join(logdir, "model.ckpt")
        saver.save(sess, checkpoint, global_step=step)
        # print('[*] Done saving checkpoint.')

    def load(self, sess, saver, logdir):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logdir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(logdir, ckpt_name))
            return True
        else:
            return False


    def generate(self, lr_image):

        sr_image = self.generator(lr_image)
        sr_image = sr_image * 255.0
        sr_image = tf.cast(sr_image, tf.int32)
        sr_image = tf.maximum(sr_image, 0)
        sr_image = tf.minimum(sr_image, 255)
        sr_image = tf.cast(sr_image, tf.uint8)

        return sr_image

    # def generate(self, lr_image):
    #
    #     sr_image = self.model(lr_image, self.ratio)
    #     sr_image = sr_image * 255.0
    #     sr_image = tf.cast(sr_image, tf.int32)
    #     sr_image = tf.maximum(sr_image, 0)
    #     sr_image = tf.minimum(sr_image, 255)
    #     sr_image = tf.cast(sr_image, tf.uint8)
    #
    #     return sr_image



