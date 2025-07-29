# src/nets/pilotNet.py

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tfkan.layers import Conv2DKAN, DenseKAN

class PilotNet(object):

    def __init__(self):
        # placeholders
        self.image_input = tf.compat.v1.placeholder(
            tf.float32, shape=[None, 66, 200, 3], name='image_input'
        )  
        self.y_        = tf.compat.v1.placeholder(
            tf.float32, shape=[None, 1], name='steering_true'
        )
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

        conv1 = Conv2DKAN(filters=8, kernel_size=5, strides=2,
                          padding='VALID', name='conv1_kan')
        h1 = tf.nn.relu(conv1(self.image_input))

        conv2 = Conv2DKAN(filters=12, kernel_size=5, strides=2,
                          padding='VALID', name='conv2_kan')
        h2 = tf.nn.relu(conv2(h1))

        conv3 = Conv2DKAN(filters=16, kernel_size=5, strides=2,
                          padding='VALID', name='conv3_kan')
        h3 = tf.nn.relu(conv3(h2))

        conv4 = Conv2DKAN(filters=24, kernel_size=3, strides=1,
                          padding='VALID', name='conv4_kan')
        h4 = tf.nn.relu(conv4(h3))

        conv5 = Conv2DKAN(filters=24, kernel_size=3, strides=1,
                          padding='VALID', name='conv5_kan')
        h5 = tf.nn.relu(conv5(h4))

        # flatten: 1×18×24 = 432
        h_flat = tf.reshape(h5, [-1, 432])

        kan_fc1 = DenseKAN(64, name='kan_fc1')
        h_fc1   = tf.nn.relu(kan_fc1(h_flat))
        h_drop  = tf.nn.dropout(h_fc1, rate=1-self.keep_prob)

        kan_fc2 = DenseKAN(1, name='kan_fc2')
        raw_out = kan_fc2(h_drop)

        # steering angle
        self.steering = tf.multiply(tf.atan(raw_out), 2.0, name='steering')
