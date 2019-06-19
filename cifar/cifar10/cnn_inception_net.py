# /usr/bin/python
# -*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np
import cifar10_input
import tensorflow.contrib.layers as layers

batch_size = 100
train_step = 100
initial_learning_rate = 0.01
datasets = 5

labels_test = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

x_images = tf.reshape(x, [-1, 32, 32, 3])

with tf.name_scope('conv-inception-1'):
    h_conv_1_1x5 = layers.conv2d(x_images, 32, [1, 5], 1, activation_fn=tf.nn.relu)
    h_conv_1_5x1 = layers.conv2d(h_conv_1_1x5, 32, [5, 1], 1, activation_fn=tf.nn.relu)

    h_conv_1_1x3 = layers.conv2d(x_images, 32, [1, 3], 1, activation_fn=tf.nn.relu)
    h_conv_1_3x1 = layers.conv2d(h_conv_1_1x3, 32, [3, 1], 1, activation_fn=tf.nn.relu)

    h_conv_1_1x1 = layers.conv2d(x_images, 32, 1, 1, activation_fn=tf.nn.relu)

    h_conv_1 = tf.concat([h_conv_1_5x1, h_conv_1_3x1, h_conv_1_1x1], 3)
    h_pool_1 = layers.max_pool2d(h_conv_1, [2, 2], stride=2, padding='SAME')  # 16

with tf.name_scope('conv-inception-2'):
    h_conv_2_1x5 = layers.conv2d(h_pool_1, 24, [1, 5], 1, activation_fn=tf.nn.relu)
    h_conv_2_5x1 = layers.conv2d(h_conv_2_1x5, 24, [5, 1], 1, activation_fn=tf.nn.relu)

    h_conv_2_1x3 = layers.conv2d(h_pool_1, 24, [1, 3], 1, activation_fn=tf.nn.relu)
    h_conv_2_3x1 = layers.conv2d(h_conv_2_1x3, 24, [3, 1], 1, activation_fn=tf.nn.relu)

    h_conv_2_1x1 = layers.conv2d(h_pool_1, 24, 1, 1, activation_fn=tf.nn.relu)

    h_conv_2 = tf.concat([h_conv_2_5x1, h_conv_2_3x1, h_conv_2_1x1], 3)
    h_pool_2 = layers.max_pool2d(h_conv_2, [2, 2], stride=2, padding='SAME')  # 8

with tf.name_scope('conv-inception-3'):
    h_conv_3_1x5 = layers.conv2d(h_pool_2, 24, [1, 5], 1, activation_fn=tf.nn.relu)
    h_conv_3_5x1 = layers.conv2d(h_conv_3_1x5, 24, [5, 1], 1, activation_fn=tf.nn.relu)

    h_conv_3_1x3 = layers.conv2d(h_pool_2, 24, [1, 3], 1, activation_fn=tf.nn.relu)
    h_conv_3_3x1 = layers.conv2d(h_conv_3_1x3, 24, [3, 1], 1, activation_fn=tf.nn.relu)

    h_conv_3_1x1 = layers.conv2d(h_pool_2, 24, 1, 1, activation_fn=tf.nn.relu)

    h_conv_3 = tf.concat([h_conv_3_5x1, h_conv_3_3x1, h_conv_3_1x1], 3)
    h_pool_3 = layers.max_pool2d(h_conv_3, [2, 2], stride=2, padding='SAME')   # 4

with tf.name_scope('conv-inception-4'):
    h_conv_4_1x5 = layers.conv2d(h_pool_3, 12, [1, 5], 1, activation_fn=tf.nn.relu)
    h_conv_4_5x1 = layers.conv2d(h_conv_4_1x5, 12, [5, 1], 1, activation_fn=tf.nn.relu)

    h_conv_4_1x3 = layers.conv2d(h_pool_3, 12, [1, 3], 1, activation_fn=tf.nn.relu)
    h_conv_4_3x1 = layers.conv2d(h_conv_4_1x3, 12, [3, 1], 1, activation_fn=tf.nn.relu)

    h_conv_4_1x1 = layers.conv2d(h_pool_3, 12, 1, 1, activation_fn=tf.nn.relu)

    h_conv_4 = tf.concat([h_conv_4_5x1, h_conv_4_3x1, h_conv_4_1x1], 3)
    h_pool_4 = layers.max_pool2d(h_conv_4, [2, 2], stride=2, padding='SAME')

with tf.name_scope('conv-inception-5'):
    h_conv_5_1x5 = layers.conv2d(h_pool_4, 6, [1, 5], 1, activation_fn=tf.nn.relu)
    h_conv_5_5x1 = layers.conv2d(h_conv_5_1x5, 6, [5, 1], 1, activation_fn=tf.nn.relu)

    h_conv_5_1x3 = layers.conv2d(h_pool_4, 6, [1, 3], 1, activation_fn=tf.nn.relu)
    h_conv_5_3x1 = layers.conv2d(h_conv_5_1x3, 6, [3, 1], 1, activation_fn=tf.nn.relu)

    h_conv_5_1x1 = layers.conv2d(h_pool_4, 6, 1, 1, activation_fn=tf.nn.relu)

    h_conv_5 = tf.concat([h_conv_5_5x1, h_conv_5_3x1, h_conv_5_1x1], 3)

y_pool = tf.reshape(h_conv_5, shape=[-1, 2*2*18])
y_pool = layers.fully_connected(y_pool, 10, activation_fn=None)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pool))

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step=global_step,
                                           decay_steps=20, decay_rate=0.9)
add_global = global_step.assign_add(1)

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

            glo_step, _, loss = sess.run([add_global, optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            rate, accur = sess.run([learning_rate, accuracy], feed_dict={x: x_test[0:100, :, :, :], y: y_test[0:100, :]})

            loss_list.append(loss)
            accur_list.append(accur)

            print('step', i, j, 'loss', loss, 'accuracy', accur, 'rate', rate)

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



