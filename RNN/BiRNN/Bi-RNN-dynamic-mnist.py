# /usr/bin/python
# -*- encoding:utf-8 -*-


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.layers import fully_connected
import numpy as np

mnist = input_data.read_data_sets('../../MNIST_data/', one_hot=True)

n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

x = tf.placeholder('float', [None, n_steps*n_input])
y = tf.placeholder('float', [None, n_classes])

x1 = tf.reshape(x, [-1, 28, 28])

lstm_fw_cell = LSTMCell(n_hidden)
lstm_bw_cell = LSTMCell(n_hidden)

outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x1, dtype=tf.float32)

output = tf.concat(outputs, 2)

pred = fully_connected(output[:, -1, :], n_classes, activation_fn=None)

cost = tf.reduce_mean(tf.reduce_sum(tf.square(pred - y)))

global_step = tf.Variable(0, trainable=False)
initial_learning_rate = 0.01

learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=3,
                                           decay_rate=0.9)

add_global = global_step.assign_add(1)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

training_epochs = 1
batch_size = 100

display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # outputs, states = sess.run([outputs, states], feed_dict={x: batch_xs, y: batch_ys})
            # print('outputs shape:', np.shape(outputs))
            # print(outputs)
            # print('states shape:', np.shape(states))
            # print(states)

            # y_pred = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys})


            # print('输出的y:\n', y_pred.shape, '\n')
            #
            # print(batch_xs.shape)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch

        if (epoch + 1) % display_step == 0:
            print('epoch= ', epoch+1, ' cost= ', avg_cost)
    print('finished')

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('test accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    # print('train accuracy: ', accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))




