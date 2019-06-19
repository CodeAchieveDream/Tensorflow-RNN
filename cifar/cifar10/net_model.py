# /usr/bin/python
# -*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cifar10_input
import init_cnn

batch_size = 100
train_step = 300
learning_rate = 0.0001

labels_test = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

x_images = tf.reshape(x, [-1, 32, 32, 3])

w_conv1 = init_cnn.weight_variable([3, 3, 3, 64])  # [-1, 32, 32, 3]
b_conv1 = init_cnn.bias_variable([64])
h_conv1 = tf.nn.relu(init_cnn.conv2d(x_images, w_conv1) + b_conv1)
h_pool1 = init_cnn.max_pool_2x2(h_conv1)


w_conv2 = init_cnn.weight_variable([3, 3, 64, 64])  # [-1, 16, 16, 64]
b_conv2 = init_cnn.bias_variable([64])
h_conv2 = tf.nn.relu(init_cnn.conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = init_cnn.max_pool_2x2(h_conv2)


w_conv3 = init_cnn.weight_variable([3, 3, 64, 32])  # [-1, 18, 8, 32]
b_conv3 = init_cnn.bias_variable([32])
h_conv3 = tf.nn.relu(init_cnn.conv2d(h_pool2, w_conv3) + b_conv3)
h_pool3 = init_cnn.max_pool_2x2(h_conv3)

w_conv4 = init_cnn.weight_variable([3, 3, 32, 16])  # [-1, 18, 8, 32]
b_conv4 = init_cnn.bias_variable([16])
h_conv4 = tf.nn.relu(init_cnn.conv2d(h_pool3, w_conv4) + b_conv4)
h_pool4 = init_cnn.max_pool_2x2(h_conv4)


w_conv5 = init_cnn.weight_variable([3, 3, 16, 10])  # [-1, 4, 4, 16]
b_conv5 = init_cnn.bias_variable([10])
h_conv5 = tf.nn.relu(init_cnn.conv2d(h_pool4, w_conv5) + b_conv5)
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
    for i in range(1):
        for j in range(train_step):
            n = np.random.randint(9890)
            # print(n)
            n = 100

            batch_x = x_train[i][n:n+batch_size, :, :, :]
            # print(batch_x.shape)
            batch_y = y_train[i][n:n+batch_size, :]
            # print(batch_y)

            _, loss = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            accur = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

            # y_ = sess.run(y_pool, feed_dict={x: batch_x, y: batch_y})
            # y__ = sess.run(y_pred, feed_dict={x: batch_x, y: batch_y})
            loss_list.append(loss)
            accur_list.append(accur)

            # print(x_.shape)
            # plt.imshow(x_[0, :, :, 1])
            # plt.axis('off')
            # plt.show()

            # print(y_[1])
            # print(y__[1])
            print('step', i, j, 'loss', loss, 'accuracy', accur)
    ind = 224
    image_label = sess.run(tf.argmax(y_pred, 1), feed_dict={x: x_train[0][ind:ind+1, :, :, :]})
    print(image_label)
    print('识别结果', labels_test[int(image_label)])
    index = int(np.argmax(y_train[0][ind:ind+1]))
    print('图片类别', labels_test[index])

    plt.imshow(x_train[0][ind, :, :, 0])
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(loss_list)), loss_list, 'r-')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(loss_list)), accur_list, 'g-')
    plt.grid(True)
    plt.show()


















































