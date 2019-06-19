# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import pickle
import io

if __name__ == '__main__':
    path = 'test'
    f = open(path, 'wb')
    data = {'a':123, 'b':'ads', 'c':[[1,2],[3,4]]}
    pickle.dump(data, f)
    f.close()

    f1 = open(path, 'rb')
    data1 = pickle.load(f1)
    print(data1)

















