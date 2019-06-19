# /usr/bin/python
# -*- encoding:utf-8 -*-


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cifar10_input
import init_cnn
from tensorflow.contrib.layers.python.layers import batch_norm

batch_size = 100
train_step = 100
learning_rate = 1e-3
datasets = 5


labels_test = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
train = tf.Variable(tf.constant(False))

x_images = tf.reshape(x, [-1, 32, 32, 3])


def batch_norm_layer(value, train=False, name='batch_norm'):
    if train is not False:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=True)
    else:
        return batch_norm(value, decay=0.9, updates_collections=None, is_training=False)


w_conv1 = init_cnn.weight_variable([3, 3, 3, 64])  # [-1, 32, 32, 3]
b_conv1 = init_cnn.bias_variable([64])
h_conv1 = tf.nn.relu(batch_norm_layer((init_cnn.conv2d(x_images, w_conv1) + b_conv1), train))
h_pool1 = init_cnn.max_pool_2x2(h_conv1)


w_conv2 = init_cnn.weight_variable([3, 3, 64, 64])  # [-1, 16, 16, 64]
b_conv2 = init_cnn.bias_variable([64])
h_conv2 = tf.nn.relu(batch_norm_layer((init_cnn.conv2d(h_pool1, w_conv2) + b_conv2), train))
h_pool2 = init_cnn.max_pool_2x2(h_conv2)


w_conv3 = init_cnn.weight_variable([3, 3, 64, 32])  # [-1, 18, 8, 32]
b_conv3 = init_cnn.bias_variable([32])
h_conv3 = tf.nn.relu(batch_norm_layer((init_cnn.conv2d(h_pool2, w_conv3) + b_conv3), train))
h_pool3 = init_cnn.max_pool_2x2(h_conv3)

w_conv4 = init_cnn.weight_variable([3, 3, 32, 16])  # [-1, 18, 8, 32]
b_conv4 = init_cnn.bias_variable([16])
h_conv4 = tf.nn.relu(batch_norm_layer((init_cnn.conv2d(h_pool3, w_conv4) + b_conv4), train))
h_pool4 = init_cnn.max_pool_2x2(h_conv4)


w_conv5 = init_cnn.weight_variable([3, 3, 16, 10])  # [-1, 4, 4, 16]
b_conv5 = init_cnn.bias_variable([10])
h_conv5 = tf.nn.relu(batch_norm_layer((init_cnn.conv2d(h_pool4, w_conv5) + b_conv5), train))
h_pool5 = init_cnn.avg_pool_4x4(h_conv5)                 # [-1, 4, 4, 10]

y_pool = tf.reshape(h_pool5, shape=[-1, 10])


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pool))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

y_pred = tf.nn.softmax(y_pool)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    x_train, y_train = cifar10_input.get_train()
    x_test, y_test = cifar10_input.get_test()
    loss_list = []
    accur_list = []
    for i in range(datasets):
        for j in range(train_step):
            n = np.random.randint(9900)
            batch_x = x_train[i][n:n+batch_size, :, :, :]
            batch_y = y_train[i][n:n+batch_size, :]

            _, loss = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y, train: True})
            accur = sess.run(accuracy, feed_dict={x: x_test[0:100, :, :, :], y: y_test[0:100, :], train: False})

            loss_list.append(loss)
            accur_list.append(accur)

            print('step', i, j, 'loss', loss, 'accuracy', accur)

    print('max accuracy :', np.max(accur))
    avg_accur = []
    for i in range(100):
        accur = sess.run(accuracy, feed_dict={x: x_test[i*100:i*100+100, :, :, :], y: y_test[i*100:i*100+100, :], train: False})
        avg_accur.append(accur)
    print('test accuracy: ', np.mean(avg_accur))

    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(np.arange(len(loss_list)), loss_list, 'r-')
    # plt.subplot(1, 2, 2)
    # plt.plot(np.arange(len(loss_list)), accur_list, 'g-')
    # plt.grid(True)
    # plt.show()









