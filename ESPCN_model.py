import tensorflow as tf
import numpy as np
import sys
import os

class ESPCN:

    def __init__(self, ratio, batch_size, lr=0):

        #self.channels = channels
        self.ratio = ratio
        self.batch_size = batch_size
        self.lr = lr

    def model(self, input, r):
        '''The structure of the network is:
                input (3 channels) ---> 3 * 3 conv (32 channels) ---> 3 * 3 conv (64 channels)
                 ---> 3 * 3 conv (48 channels)
                Where `conv` is 2d convolutions with a non-linear activation (tanh) at the output.
        '''
        init = tf.contrib.layers.xavier_initializer()
        x = tf.layers.conv2d(input, 64, 3, strides=(1, 1),
                             padding='same', kernel_initializer=init, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 256, 3, strides=(1, 1),
                             padding='same', kernel_initializer=init, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 128, 3, strides=(1, 1),
                             padding='same', kernel_initializer=init, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 48, 3, strides=(1, 1),
                             padding='same', kernel_initializer=init, activation=tf.nn.tanh)
        print(x.get_shape())
        output = self.PS(x, r, color=True)

        return output

    def loss(self, output, target, type='L2'):
        if type=='L2':
            residual = output - target
            square = tf.square(residual)
            reduce_loss = tf.reduce_mean(square)
            tf.summary.scalar('loss', reduce_loss)
        else:
            residual = output - target
            square = tf.abs(residual)
            reduce_loss = tf.reduce_mean(square)
            tf.summary.scalar('loss', reduce_loss)
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

    def PS(self, X, r, color=False):
        # Main OP that you can arbitrarily use in you tensorflow code
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

        sr_image = self.model(lr_image, self.ratio)
        sr_image = sr_image * 255.0
        sr_image = tf.cast(sr_image, tf.int32)
        sr_image = tf.maximum(sr_image, 0)
        sr_image = tf.minimum(sr_image, 255)
        sr_image = tf.cast(sr_image, tf.uint8)

        return sr_image



