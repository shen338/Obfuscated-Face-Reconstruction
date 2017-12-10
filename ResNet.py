import tensorflow as tf
import numpy as np
from VGG import *
import sys


class ResNet:

    def __init__(self, lr, bottleneck_num, batchsize, ratio):
        self.ratio = ratio
        self.batch_size = batchsize
        self.learning_rate = lr
        self.bottlenecks = bottleneck_num

    def generator(self, inputs, bottleneck_num):

        w_init = tf.random_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        g_init = tf.random_normal_initializer(1., 0.02)

        inputs = tf.layers.conv2d(inputs, 64, 3, strides=(1, 1),
                             padding='same', kernel_initializer=w_init,
                             activation=self.Prelu, name='conv1')
        temp = inputs
        for ii in range(1, bottleneck_num + 1):
            block = tf.layers.conv2d(inputs, 64, 3, strides=(1, 1),
                                      padding='same', kernel_initializer=w_init,
                                      activation=self.Prelu,
                                     name='B%s_conv1'%ii)
            block = tf.layers.batch_normalization(block, name='B%s_bn1'%ii)

            block = self.Prelu(block)

            block = tf.layers.conv2d(block, 64, 3, strides=(1, 1),
                                     padding='same', kernel_initializer=w_init,
                                     activation=self.Prelu,
                                     name='B%s_conv2' %ii)

            block = tf.layers.batch_normalization(block, name='B%s_bn2' %ii)

            inputs = inputs + block

        inputs = tf.layers.conv2d(inputs, 64, 3, strides=(1, 1),
                                 padding='same', kernel_initializer=w_init,
                                 activation=self.Prelu,
                                 name='conv3')
        inputs = tf.layers.batch_normalization(inputs, name='bn3')

        inputs = inputs + temp

        inputs = tf.layers.conv2d(inputs, 256, 3, strides=(1,1),
                                  padding='same', kernel_initializer=w_init,
                                 activation=self.Prelu,
                                 name='conv4')

        inputs = self.PS(inputs, scale=2, name ='PixelShuffler1')
        inputs = self.Prelu(inputs)

        inputs = tf.layers.conv2d(inputs, 256, 3, strides=(1, 1),
                                  padding='same', kernel_initializer=w_init,
                                  activation=self.Prelu,
                                  name='conv5')

        inputs = self.PS(inputs, scale=2, name ='PixelShuffler2')
        inputs = self.Prelu(inputs)

        outputs = tf.layers.conv2d(inputs, 3, 9, strides=(1,1),
                                   padding='same', kernel_initializer=w_init,
                                   activation=None,
                                   name='conv6')
        return outputs

    def content_loss(self, y, target, para_dict):
        vgg_y = Vgg16(y, para_dict)
        vgg_target = Vgg16(target, para_dict)

        loss = 0

        loss = loss + tf.reduce_mean(tf.square(vgg_y.conv1_2 - vgg_target.conv1_2))

        loss = loss + tf.reduce_mean(tf.square(vgg_y.conv2_2 - vgg_target.conv2_2))

        loss = loss + tf.reduce_mean(tf.square(vgg_y.conv3_3 - vgg_target.conv3_3))

        loss = loss + tf.reduce_mean(tf.square(vgg_y.conv4_3 - vgg_target.conv4_3))

        loss = loss + tf.reduce_mean(tf.square(vgg_y.conv5_3 - vgg_target.conv5_3))

        tf.summary.scalar('content_loss', loss)

        return loss

    def MSE_loss(self, y, target):

        residual = y - target
        square = tf.square(residual)
        reduce_loss = tf.reduce_mean(square)
        tf.summary.scalar('MSE_loss', reduce_loss)

        return reduce_loss

    def abs_loss(self, y, target):
        residual = y - target
        square = tf.abs(residual)
        reduce_loss = tf.reduce_mean(square)
        tf.summary.scalar('abs_loss', reduce_loss)

        return reduce_loss

    def Prelu(self, input, name='Prelu'):

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            alpha = tf.get_variable('Prelu_alpha', input.get_shape()[-1],
                  initializer=tf.constant_initializer(value=0.0), dtype=tf.float32)

        output = tf.nn.relu(input) + alpha*(input - tf.abs(input))*0.5

        return output

    def _phase_shift(self, I, scale):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
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

    def generate(self, lr_image, bottleneck_num):

        sr_image = self.generator(lr_image, bottleneck_num)
        sr_image = sr_image * 255.0
        sr_image = tf.cast(sr_image, tf.int32)
        sr_image = tf.maximum(sr_image, 0)
        sr_image = tf.minimum(sr_image, 255)
        sr_image = tf.cast(sr_image, tf.uint8)

        return sr_image






# a = ResNet(1,1,1,1)
# b = tf.constant(-1, shape=(50,64,64,64), dtype=tf.float32)
# p = a.Prelu(b)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(p))
# print(p)
