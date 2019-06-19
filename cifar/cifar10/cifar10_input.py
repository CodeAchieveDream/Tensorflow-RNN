# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import pickle
import random
import os
import matplotlib.pyplot as plt


path = 'D:\\tmp\cifar10_data\cifar-10-batches-py'

labels_test = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test = 'test_batch'


def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data


def image_reshape(images):
    images = images.reshape([-1, 3, 32, 32])
    images = np.swapaxes(images, 1, 2)
    images = np.swapaxes(images, 2, 3)
    # images = images/255
    return images


def one_hot(labels):
    labels_onehot = np.zeros([10000, 10])
    for i, value in enumerate(labels):
        labels_onehot[i, value] = 1
    return labels_onehot


def get_train():
    x = []
    y = []
    for i, dir in enumerate(train_list):
        file = os.path.join(path, dir)
        # print(file)
        data = load(file)
        images = np.array(data[b'data'], dtype=float)
        images = image_reshape(images)
        labels = np.array(data[b'labels'])
        # print(labels[0])
        labels = one_hot(labels)
        x.append(images)
        y.append(labels)
    return x, y


def get_test():
    file = os.path.join(path, test)
    data = load(file)

    images = np.array(data[b'data'], dtype=float)
    images = image_reshape(images)

    labels = np.array(data[b'labels'])
    labels = one_hot(labels)
    return images, labels


if __name__ == '__main__':

    x, y = get_test()

    # print(x[0][1, :, :, 1])
    # print(y[0].shape)
    # print(y[0][0])
    # print(y[1][0])
    # print(y[2][0])
    # print(y[3][0])
    # print(y[4][0])


















