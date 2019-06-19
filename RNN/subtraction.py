# /usr/bin/python
# -*- encoding:utf-8 -*-

import copy, numpy as np

np.random.seed(0)


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output*(1-output)


int2binary = {}
binary_dim = 8

# 计算0-256的二进制
largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)







