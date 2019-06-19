# /usr/bin/python
# -*- encoding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import GRUCell

tf.reset_default_graph()

x = np.random.randn(2, 4, 5)

print(x)

# x[1, 1:] = 0

print(x)

seq_lengths = [4, 4]

#分别建立一个lstm和gru的cell，比较输出的状态
cell = BasicLSTMCell(num_units=3, state_is_tuple=True)
gru = GRUCell(3)

outputs, last_states, = tf.nn.dynamic_rnn(cell, x, seq_lengths, dtype=tf.float64)

gruoutput, grulast_states = tf.nn.dynamic_rnn(gru, x, seq_lengths, dtype=tf.float64)

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

result, sta, gruout, grusta = sess.run([outputs, last_states, gruoutput, grulast_states])

print('全序列：\n', result)
print('短序列：\n', result[1])

print('lstm的状态：\n', len(sta), '\n', sta)


print('gru短序列: \n', gruout)
print('gru的状态: \n', len(grusta), '\n', grusta)




































