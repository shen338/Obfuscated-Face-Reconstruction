import tensorflow as tf
import numpy as np
import sys
import os

class WGAN:

    def __init__(self, gamma):

        self.gamma = gamma

    def generator(self, input, bottleneck_num, reuse=False):

        with tf.variable_scope('generator') as scope:

            if reuse:
                scope.reuse_variables()

            w_init = tf.random_normal_initializer(stddev=0.02)
            b_init = tf.constant_initializer(value=0.0)
            g_init = tf.random_normal_initializer(1., 0.02)

            inputs = tf.layers.conv2d(input, 64, 3, strides=(1, 1),
                                      padding='same', kernel_initializer=w_init,
                                      name='conv1')
            temp = inputs
            for ii in range(1, bottleneck_num + 1):
                block = tf.layers.conv2d(inputs, 64, 3, strides=(1, 1),
                                         padding='same', kernel_initializer=w_init,
                                         name='B%s_conv1' % ii)
                block = tf.layers.batch_normalization(block, name='B%s_bn1' % ii)

                block = self.Prelu(block, 'Prelu_%s'%ii)

                block = tf.layers.conv2d(block, 64, 3, strides=(1, 1),
                                         padding='same', kernel_initializer=w_init,
                                         name='B%s_conv2' % ii)

                block = tf.layers.batch_normalization(block, name='B%s_bn2' % ii)

                inputs = inputs + block

            inputs = tf.layers.conv2d(inputs, 64, 3, strides=(1, 1),
                                      padding='same', kernel_initializer=w_init,
                                      name='conv3')
            inputs = tf.layers.batch_normalization(inputs, name='bn3')

            inputs = inputs + temp

            inputs = tf.layers.conv2d(inputs, 256, 3, strides=(1, 1),
                                      padding='same', kernel_initializer=w_init,
                                      name='conv4')

            inputs = self.PS(inputs, scale=2, name='PixelShuffler1')
            inputs = self.Prelu(inputs, name='Prelu_PS1')

            inputs = tf.layers.conv2d(inputs, 256, 3, strides=(1, 1),
                                      padding='same', kernel_initializer=w_init,
                                      name='conv5')

            inputs = self.PS(inputs, scale=2, name='PixelShuffler2')
            inputs = self.Prelu(inputs, name="Prelu_PS2")

            outputs = tf.layers.conv2d(inputs, 3, 5, strides=(1, 1),
                                       padding='same', kernel_initializer=w_init,
                                       activation=None,
                                       name='conv6')

            print(outputs.get_shape(), 'test1')

            return outputs

    def Prelu(self, input, name='Prelu'):

        with tf.variable_scope(name):
            alpha = tf.get_variable('Prelu_alpha', input.get_shape()[-1],
                  initializer=tf.constant_initializer(value=0.0), dtype=tf.float32)

        output = tf.nn.relu(input) + alpha*(input - tf.abs(input))*0.5

        return output

    def discrimintor(self, input, reuse=False):

        with tf.variable_scope('discriminator') as scope:

            if reuse:
                scope.reuse_variables()

            init = tf.contrib.layers.xavier_initializer()

            # 32x32
            x = tf.layers.conv2d(input, 128, 3, strides=(2, 2),
                                 padding='same', kernel_initializer=init,
                                 activation=self.leaky_relu, name='D_downsample_2')
            #x = tf.layers.batch_normalization(x)
            # 16x16
            x = tf.layers.conv2d(x, 256, 3, strides=(2, 2),
                                 padding='same', kernel_initializer=init,
                                 activation=self.leaky_relu, name='D_downsample_3')
            #x = tf.layers.batch_normalization(x)
            # 8x8
            x = tf.layers.conv2d(x, 512, 3, strides=(2, 2),
                                 padding='same', kernel_initializer=init,
                                 activation=self.leaky_relu, name='D_downsample_4')
            #x = tf.layers.batch_normalization(x)
            # 4x4
            x = tf.layers.conv2d(x, 512, 3, strides=(2, 2),
                                 padding='same', kernel_initializer=init,
                                 activation=self.leaky_relu, name='D_downsample_5')
            #x = tf.layers.batch_normalization(x)
            # 1x1
            x = tf.layers.conv2d(x, 512, 3, strides=(2, 2),
                                 padding='SAME', kernel_initializer=init,
                                 activation=self.leaky_relu, name='D_downsample_6')
            x = tf.layers.batch_normalization(x)


            print(x.get_shape())
            x = tf.reshape(x, [-1,512*2*2])
            print(x.get_shape())

            x = tf.layers.dense(x, 200)
            x = self.leaky_relu(x)
            x = tf.layers.dense(x, 1)

            return x

    def gan_loss(self, hr_image, D_hr_image, G_image, D_G_image, Kt):

        G_loss = self.loss(hr_image, G_image, type='L1')
        D_real = self.loss(hr_image, D_hr_image, type='L1')
        D_fake = self.loss(G_image, D_G_image, type='L1')
        D_loss = D_real - Kt*D_fake

        W_loss = D_real + tf.abs(self.gamma*D_real - G_loss)

        return G_loss, D_loss, W_loss

    def leaky_relu(self, x, alpha=0.02):
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

    def _phase_shift(self, I, scale):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        #print(I.get_shape())
        X = tf.reshape(I, (-1, a, b, scale, scale))
        X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (-1, a * scale, b * scale, 1))

    def PS(self, X, scale=2, name = None):
        # Main OP that you can arbitrarily use in you tensorflow code
        with tf.name_scope(name):
            channels = int(X.get_shape()[-1])
            multi = scale*scale

            Xc = tf.split(X, int(channels/multi), axis=3)
            X = tf.concat([self._phase_shift(x, scale) for x in Xc], 3)
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

        sr_image = self.generator(lr_image, bottleneck_num=2)
        sr_image = (sr_image + 1)*127.5
        sr_image = tf.cast(sr_image, tf.int32)
        sr_image = tf.maximum(sr_image, 0)
        sr_image = tf.minimum(sr_image, 255)
        sr_image = tf.cast(sr_image, tf.uint8)

        return sr_image


