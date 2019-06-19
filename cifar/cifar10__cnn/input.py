# /usr/bin/python
# -*- encoding:utf-8 -*-


import pickle
import numpy as np
import random


def load(file_name):
    with open(file_name, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        return data


def get_train():
    data1 = load('D:\\tmp\cifar10_data\cifar-10-batches-py\data_batch_1')
    x1 = np.array(data1[b'data'])
    x1 = x1.reshape(-1, 32, 32, 3)
    y1 = np.array(data1[b'labels'])
    data2 = load('D:\\tmp\cifar10_data\cifar-10-batches-py\data_batch_2')
    x2 = np.array(data2[b'data'])
    x2 = x2.reshape(-1, 32, 32, 3)
    y2 = np.array(data2[b'labels'])
    train_data = np.r_[x1, x2]
    train_labels = np.r_[y1, y2]
    data3 = load('D:\\tmp\cifar10_data\cifar-10-batches-py\data_batch_3')
    x3 = np.array(data3[b'data'])
    x3 = x3.reshape(-1, 32, 32, 3)
    y3 = data3[b'labels']
    train_data = np.r_[train_data, x3]
    train_labels = np.r_[train_labels, y3]
    data4 = load('D:\\tmp\cifar10_data\cifar-10-batches-py\data_batch_4')
    x4 = np.array(data4[b'data'])
    x4 = x4.reshape(-1, 32, 32, 3)
    y4 = data4[b'labels']
    train_data = np.r_[train_data, x4]
    train_labels = np.r_[train_labels, y4]
    return list(train_data), list(train_labels)


def get_test():
    data1 = load('D:\\tmp\cifar10_data\cifar-10-batches-py\\test_batch')
    x = np.array(data1[b'data'])
    x = x.reshape(-1, 32, 32, 3)
    y = data1[b'labels']
    return list(x), list(y)


def get_batch(batch_size, image, label):
    batch_image = list()
    batch_label = list()
    indexs = list()
    for i in range(batch_size):
        index = random.randint(0, len(image) - 1)
        while index in indexs:
            index = random.randint(0, len(image) - 1)
        d = list(image[index])
        batch_image.append(d)
        z = label[index]
        batch_label.append(z)
        indexs.append(index)
    return batch_image, batch_label










