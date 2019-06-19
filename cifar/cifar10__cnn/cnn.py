# /usr/bin/python
# -*- encoding:utf-8 -*-


import tensorflow as tf
import input
import numpy as np
import init
import matplotlib.pyplot as plt
import random

max_steps = 5000
batch_size = 128
display = 100


def LRnorm(tensor):
    return tf.nn.lrn(tensor, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


def accuracy(test_labels, test_y_out):
    test_labels = tf.to_int64(test_labels)
    prediction_result = tf.equal(test_labels, tf.argmax(y_, 1))
    accu = tf.reduce_mean(tf.cast(prediction_result, tf.float32))
    return accu


# train_image, train_label = cifar10_input.distorted_inputs(batch_size=batch_size, data_dir="cifar-10-batches-bin")
# test_image, test_label = cifar10_input.inputs(batch_size=batch_size, data_dir="cifar-10-batches-bin", eval_data=True)

with tf.name_scope('Input'):
    image = tf.placeholder('float', [batch_size, 32, 32, 3])
    label = tf.placeholder('float', [batch_size])

with tf.name_scope('ConLayer_1'):
    we1 = init.weight_init([5, 5, 3, 32], 0.05)
    b1 = init.bias_init([32])
    conv1 = tf.nn.relu(init.conv2d(image, we1) + b1)
    pool1 = init.max_pool(conv1)
    LRn1 = LRnorm(pool1)

with tf.name_scope('ConLayer_2'):
    w2 = init.weight_init([5, 5, 32, 32], 0.05)
    b2 = init.bias_init([32])
    conv2 = tf.nn.relu(init.conv2d(LRn1, w2) + b2)
    LRn2 = LRnorm(conv2)
    pool2 = init.max_pool(LRn2)

with tf.name_scope('FullLayer_1'):
    reshape = tf.reshape(pool2, [batch_size, -1])
    n_input = reshape.get_shape()[1].value
    w3 = init.l2_weight_init([n_input, 128], 0.05, w1=0.001)
    b3 = init.bias_init([128])
    full_1 = tf.nn.relu(tf.matmul(reshape, w3) + b3)

with tf.name_scope("FullLayer_2"):
    w4 = init.l2_weight_init([128, 64], 0.05, w1=0.003)
    b4 = init.bias_init([64])
    full_2 = tf.nn.relu(tf.matmul(full_1, w4) + b4)

with tf.name_scope('Inference'):
    w5 = init.weight_init([64, 10], 1 / 96.0)
    b5 = init.bias_init([10])
    logits = tf.add(tf.matmul(full_2, w5), b5)
    y_ = tf.nn.softmax(logits)

with tf.name_scope('Loss'):
    label = tf.cast(label, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

top_k_op = tf.nn.in_top_k(logits, label, 1)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# tf.train.start_queue_runners(sess=sess)

Cross_loss = []
print("start")
train_image, train_label = input.get_train()

for i in range(5000):
    # batch_images, batch_labels = sess.run([train_image, train_label])           #用session读取数据效率低，改成python读取
    batch_images, batch_labels = input.get_batch(batch_size, train_image, train_label)
    _, cross_entropy = sess.run([train_op, loss], feed_dict={image: batch_images, label: batch_labels})
    Cross_loss.append(cross_entropy)
    if i % display == 0:
        print('epoch', i, 'loss:', cross_entropy)

test_image, test_label = input.get_test()
for i in range(10):
    test_batch_image, test_batch_label = input.get_batch(batch_size, test_image, test_label)
    ys = sess.run([top_k_op], feed_dict={image: test_batch_image, label: test_batch_label})
    print(np.sum(ys) / batch_size)

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(Cross_loss)
plt.grid()
plt.title('Train loss')












