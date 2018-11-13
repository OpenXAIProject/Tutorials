import os
import re
import numpy as np
import tensorflow as tf

from config import Config

class Network(object):
    def __init__(self, device, model_name, num_actions):
        self.device = device 
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self.create_placeholder()
                self.create_network()
                # self.create_train_op()
                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())

                vars = tf.trainable_variables()
                self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)

    def create_placeholder(self):
        self.x = tf.placeholder(
            tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')

    def create_network(self):
        # As implemented in A3C paper
        self.n1 = self.conv2d_layer(self.x, 8, 16, 'conv11', strides=[1, 4, 4, 1])
        self.n2 = self.conv2d_layer(self.n1, 4, 32, 'conv12', strides=[1, 2, 2, 1])
        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])
        _input = self.n2

        flatten_input_shape = _input.get_shape()
        nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]

        self.flat = tf.reshape(_input, shape=[-1, nb_elements._value])
        self.d1 = self.dense_layer(self.flat, 256, 'dense1')

        self.logits_v = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v', func=None), axis=[1])
        self.logits_p = self.dense_layer(self.d1, self.num_actions, 'logits_p', func=None)
        self.softmax_p = tf.nn.softmax(self.logits_p)

    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.matmul(input, w) + b
            if func is not None:
                output = func(output)

        return output

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    def predict_p_and_v_single(self, x):
        p, v = self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x[np.newaxis, :]})
        return p[0], v[0]

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (self.model_name, episode)

    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[2])

    # def load(self):
    #     filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
    #     print(filename)
    #     self.saver.restore(self.sess, filename)
    #     return self._get_episode_from_filename(filename)