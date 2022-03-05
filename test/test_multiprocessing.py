#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd() + '/src')
import numpy as np
import multiprocessing as mp


x = np.array([[1,2], [3,4], [5,6], [7,8]])
y = np.array([[-1,-2], [-3,-4], [-5,-6], [-7,-8]])


def f(idx):
    return x[idx, :]+y[idx, :], x[idx, :]-y[idx, :]


class A():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def cal(self, idx):
        return self.x[idx, :]+self.y[idx, :], self.x[idx, :]-self.y[idx, :]
    def run(self):
        t = mp.Pool()
        rs = t.map(self.cal, np.arange(0, x.shape[0]))
        t.close()
        return rs

if __name__ == '__main__':
    print("# cpu cores: ", mp.cpu_count())

    print("x: ", x)
    print("y: ", y)

    # process_pool = mp.Pool()

    # results = process_pool.map(f, np.arange(0, x.shape[0]))

    results =  mp.Pool().map(f, np.arange(0, x.shape[0]))


    a = A(x, y)
    rs = a.run()
    print("rs: ", rs)


    print(results)
