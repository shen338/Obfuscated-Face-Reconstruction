

import tensorflow as tf
import numpy as np

class AlexNet:

    def __init__(self,x, keep_prob, num_classes, skip_layer,
               weights_path = 'DEFAULT'):
        self.Drop_rate = keep_prob
        self.num_class = num_classes
        self.skip_layer = skip_layer
        self.input = x
        if weights_path == 'DEFAULT':
            self.path = 'bvlc_alexnet.npy'
        else:
            self.path = weights_path
        self.create()
    def create(self):
        # 1st conv
        conv1 = conv2d(self.input, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = batchnorm(conv1, 2, 2e-05, 0.75, name='norm1')
        pool1 = max_pooling(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu) -> Lrn -> Poolwith 2 groups
        conv2 = conv2d(pool1, 5, 5, 256, 1, 1, group = 2, name='conv2')
        norm2 = batchnorm(conv2, 2, 2e-05, 0.75, name='norm2')
        pool2 = max_pooling(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv2d(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv2d(conv3, 3, 3, 384, 1, 1, group = 2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv2d(conv4, 3, 3, 256, 1, 1, group = 2, name='conv5')
        pool5 = max_pooling(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.Drop_rate)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.Drop_rate)

        # 8th Layer: FC and return unscaled activations
        # (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = fc(dropout7, 4096, self.num_class, relu=False, name='fc8')


    def load_pretrained_alexnet(self, session):

        weights_dict = np.load(self.path, encoding='bytes').item()
        #print(weights_dict)
        for layer_name in weights_dict:
            print(layer_name)
            if layer_name not in self.skip_layer:
                with tf.variable_scope(layer_name, reuse=True):
                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[layer_name]:

                        # Biases
                        if len(data.shape) == 1:

                            var = tf.get_variable('bias', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:

                            var = tf.get_variable('weight', trainable=False)
                            session.run(var.assign(data))


def conv2d(x,filter_height, filter_width, num_filters, stride_x, stride_y, name,
               padding='SAME', group = 1):
    input_channel = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)
    # initial weight and bias for conv layer
    with tf.variable_scope(name) as scope:  ## !!!
        W = tf.get_variable('weight', shape=[filter_height, filter_width,
                                              input_channel/group, num_filters])
        b = tf.get_variable('bias', shape=[num_filters])

        if group == 1:
            conv = convolve(x, W)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=group, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=group, value=W)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        biased = tf.reshape(tf.nn.bias_add(conv, b), conv.shape)  ## !!!

        relu = tf.nn.relu(biased, name=scope.name)
        return relu

def fc(x,input_num,output_num,name,relu = True):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weight', shape=[input_num, output_num])
        b = tf.get_variable('bias', shape=[output_num])

        act = tf.nn.xw_plus_b(x, W, b, name=scope.name)

        if relu == True:
            # Apply ReLu
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

def max_pooling(x, filter_height, filter_width, stride_x, stride_y, name, padding = 'SAME'):
    return tf.nn.max_pool(x,ksize=[1,filter_height,filter_width,1],strides=[1,stride_y,stride_x,1],
                              padding = padding, name = name)

def batchnorm(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius = radius,
                                            alpha = alpha, beta = beta,
                                            bias = bias, name = name)
def dropout(x, drop_rate):
    return tf.nn.dropout(x, drop_rate)



