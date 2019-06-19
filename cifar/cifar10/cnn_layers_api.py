# /usr/bin/python
# -*- encoding:utf-8 -*-


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cifar10_input
import tensorflow.contrib.layers as layers

batch_size = 100
train_step = 100
learning_rate = 1e-3
datasets = 1


labels_test = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

x_images = tf.reshape(x, [-1, 32, 32, 3])

h_conv1 = layers.conv2d(x_images, 64, 3, 1, activation_fn=tf.nn.relu)
h_pool1 = layers.max_pool2d(h_conv1, [2, 2], stride=2, padding='SAME')

h_conv2 = layers.conv2d(h_pool1, 64, 3, 1, activation_fn=tf.nn.relu)
h_pool2 = layers.max_pool2d(h_conv2, [2, 2], stride=2, padding='SAME')

h_conv3 = layers.conv2d(h_pool2, 32, 3, 1, activation_fn=tf.nn.relu)
h_pool3 = layers.max_pool2d(h_conv3, [2, 2], stride=2, padding='SAME')

h_conv4 = layers.conv2d(h_pool3, 16, 3, 1, activation_fn=tf.nn.relu)
h_pool4 = layers.max_pool2d(h_conv4, [2, 2], stride=2, padding='SAME')

h_conv5 = layers.conv2d(h_pool4, 10, 3, 1, activation_fn=tf.nn.relu)
y_pool = tf.reshape(h_conv5, shape=[-1, 40])

y_pool = layers.fully_connected(y_pool, 10, activation_fn=None)

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

            _, loss = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            accur = sess.run(accuracy, feed_dict={x: x_test[0:100, :, :, :], y: y_test[0:100, :]})

            loss_list.append(loss)
            accur_list.append(accur)

            print('step', i, j, 'loss', loss, 'accuracy', accur)

    print('max accuracy :', np.max(accur))
    avg_accur = []
    for i in range(100):
        accur = sess.run(accuracy, feed_dict={x: x_test[i*100:i*100+100, :, :, :], y: y_test[i*100:i*100+100, :]})
        avg_accur.append(accur)
    print('test accuracy: ', np.mean(avg_accur))

    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(np.arange(len(loss_list)), loss_list, 'r-')
    # plt.subplot(1, 2, 2)
    # plt.plot(np.arange(len(loss_list)), accur_list, 'g-')
    # plt.grid(True)
    # plt.show()









