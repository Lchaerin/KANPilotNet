# src/nets/pilotNet.py

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# KAN dense layer import
from tfkan.layers import DenseKAN

def _weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def _conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

class PilotNet(object):
    """End-to-End PilotNet to output steering angles given input images"""

    def __init__(self):
        # placeholders
        self.image_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=[None, 66, 200, 3],
                                                   name='image_input')
        self.y_          = tf.compat.v1.placeholder(tf.float32,
                                                   shape=[None, 1],
                                                   name='steering_true')
        self.keep_prob   = tf.compat.v1.placeholder(tf.float32,
                                                   name='keep_prob')

        # conv1
        W_conv1 = _weight_variable([5, 5, 3, 24])
        b_conv1 = _bias_variable([24])
        h_conv1 = tf.nn.relu(_conv2d(self.image_input, W_conv1, 2) + b_conv1)

        # conv2
        W_conv2 = _weight_variable([5, 5, 24, 36])
        b_conv2 = _bias_variable([36])
        h_conv2 = tf.nn.relu(_conv2d(h_conv1, W_conv2, 2) + b_conv2)

        # conv3
        W_conv3 = _weight_variable([5, 5, 36, 48])
        b_conv3 = _bias_variable([48])
        h_conv3 = tf.nn.relu(_conv2d(h_conv2, W_conv3, 2) + b_conv3)

        # conv4
        W_conv4 = _weight_variable([3, 3, 48, 64])
        b_conv4 = _bias_variable([64])
        h_conv4 = tf.nn.relu(_conv2d(h_conv3, W_conv4, 1) + b_conv4)

        # conv5
        W_conv5 = _weight_variable([3, 3, 64, 64])
        b_conv5 = _bias_variable([64])
        h_conv5 = tf.nn.relu(_conv2d(h_conv4, W_conv5, 1) + b_conv5)

        # fully connected → DenseKAN
        h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])

        # KAN-FC1
        kan_fc1 = DenseKAN(1164, name='kan_fc1')
        h_fc1 = tf.nn.relu(kan_fc1(h_conv5_flat))
        h_fc1_drop = tf.nn.dropout(h_fc1, rate=1-self.keep_prob)

        # KAN-FC2
        kan_fc2 = DenseKAN(50, name='kan_fc2')
        h_fc2 = tf.nn.relu(kan_fc2(h_fc1_drop))
        h_fc2_drop = tf.nn.dropout(h_fc2, rate=1-self.keep_prob)

        kan_fc3 = DenseKAN(1, name='kan_fc3')
        raw_output = kan_fc3(h_fc2_drop)
        self.steering = tf.multiply(tf.atan(raw_output), 2.0, name='steering')
