# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from PIL import Image

path1 = 'D:\\tmp\cifar10_data\cifar-10-batches-py\data_batch_1'
path2 = 'D:\\tmp\cifar10_data\cifar-10-batches-py\data_batch_2'
path3 = 'D:\\tmp\cifar10_data\cifar-10-batches-py\data_batch_3'
path4 = 'D:\\tmp\cifar10_data\cifar-10-batches-py\data_batch_4'
path5 = 'D:\\tmp\cifar10_data\cifar-10-batches-py\data_batch_5'

path6 = 'D:\\tmp\cifar10_data\cifar-10-batches-py\\test_batch'

if __name__ == '__main__':
    with open(path1, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

        # print(data[b'batch_label'])
        # print(data[b'labels'])
        # print(data[b'data'])
        # print(data[b'filenames'])

        print(data[b'data'].shape)

        images_batch = np.array(data[b'data'])
        images = images_batch.reshape([-1, 3, 32, 32])
        images = np.swapaxes(images, 1, 2)
        images = np.swapaxes(images, 2, 3)

        n = 105

        print(images.shape)
        imgs = images[n, :, :, :]
        # img = np.stack((imgs[:, 0, :], imgs[:, 1, :], imgs[:, 2, :]), 1)
        print(imgs.shape)
        print(data[b'labels'][n])

        plt.imshow(imgs[:, :, 1])
        plt.axis('off')
        plt.show()



























