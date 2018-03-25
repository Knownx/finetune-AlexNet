# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   create time: 2018.03.14 Wed. 23h45m16s
   author: Chuanfeng Liu
   e-mail: microlj@126.com
   github: https://github.com/Knownx
'''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
import numpy as np

class AlexNet(object):
    def __init__(self, input, keep_prob, numClasses, skipLayer, pretrained='bvlc_alexnet.npy'):
        self.input = input
        self.keep_prob = keep_prob
        self.numClasses = numClasses
        self.skipLayer = skipLayer
        self.pretrained = pretrained

        self.Net()

    def Net(self):
        """Implementation of original AlexNet"""
        # 1st layer
        conv1 = convLayer(self.input, 11, 11, 96, [4,4], padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = maxPool(norm1, 3, 3, [2,2], padding='VALID', name='pool1')

        # 2nd layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = convLayer(pool1, 5, 5, 256, [1,1], groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = maxPool(norm2, 3, 3, [2,2], padding='VALID', name='pool2')

        # 3rd Layer
        conv3 = convLayer(pool2, 3, 3, 384, [1,1], name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = convLayer(conv3, 3, 3, 384, [1,1], groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = convLayer(conv4, 3, 3, 256, [1,1], groups=2, name='conv5')
        pool5 = maxPool(conv5, 3, 3, [2,2], padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fcLayer(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.keep_prob)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fcLayer(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.keep_prob)

        # 8th Layer: FC and return unscaled activations
        self.fc8 = fcLayer(dropout7, 4096, self.numClasses, relu=False, name='fc8')

    def pretrainedModel(self, session):
        """Implementation of loading pretrained parameters"""
        weightsDict = np.load(self.pretrained, encoding='bytes').item()
        for item in weightsDict:
            #print ('-'+item)
            if item not in self.skipLayer:
                with tf.variable_scope(item, reuse=True):
                    for data in weightsDict[item]:
                        #print('---' + str(data))
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

def convLayer(input, filterHeight, filterWidth, depth, strides, name, padding='SAME', groups=1):
    """Implementation of convolution layer"""
    input_channels = int(input.get_shape()[-1])
    convolve = lambda i,j: tf.nn.conv2d(i, j, strides=[1, strides[0], strides[1], 1], padding=padding)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filterHeight, filterWidth, input_channels/groups, depth])
        biases = tf.get_variable('biases', shape=[depth])

    if groups == 1:
        conv = convolve(input, weights)
    else:
        input_groups = tf.split(value=input, num_or_size_splits=groups, axis=3)
        weight_groups = tf.split(value=weights, num_or_size_splits=groups, axis=3)
        output_groups = [convolve(i,j) for i,j in zip(input_groups, weight_groups)]

        # Concat the output together
        conv = tf.concat(values=output_groups, axis=3)

    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu

def maxPool(input, filterHeight, filterWidth, strides, name, padding='SAME'):
    """Implementation of max-pooling"""
    return tf.nn.max_pool(input, ksize=[1, filterHeight, filterWidth, 1], strides=[1, strides[0], strides[1], 1], padding=padding, name=name)

def fcLayer(input, in_size, out_size, name, relu=True):
    """Implementation of fully connected layer"""
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[in_size, out_size], trainable=True)
        biases = tf.get_variable('biases', shape=[out_size], trainable=True)
        output = tf.nn.xw_plus_b(input, weights, biases, name=scope.name)

        if relu:
            return tf.nn.relu(output)
        else:
            return output

def dropout(input, keep_prob):
    """Implementation of dropout layer"""
    return tf.nn.dropout(input, keep_prob)

def lrn(input, radius, alpha, beta, name, bias=1.0):
    """Implementation of local response normalization layer"""
    return tf.nn.local_response_normalization(input, depth_radius=radius, alpha=alpha, beta=beta, name=name, bias=bias)
