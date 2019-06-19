# /usr/bin/python
# -*- encoding:utf-8 -*-


import tensorflow as tf

def l2_weight_init(shape, stddev, w1):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(weight), w1, name="weight_loss")
        tf.add_to_collection("losses", weight_loss)
    return weight


def weight_init(shape, stddev):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    return weight


def bias_init(shape):
    return tf.Variable(tf.random_normal(shape))


def conv2d(image, weight):
    return tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="SAME")


def max_pool(tensor):
    return tf.nn.max_pool(tensor, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")










