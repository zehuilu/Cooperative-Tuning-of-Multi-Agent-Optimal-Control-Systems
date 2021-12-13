#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import math

def phi(x, y, points: list):
    phi = 0.0
    for ele in points:
        phi += np.exp(-(x-ele[0])**2 - (y-ele[1])**2)

    return phi

if __name__ == '__main__':
    points = [[1,1], [1.5,1], [2,1.5]]

    x = np.linspace(-1, 3.5, 100)
    y = np.linspace(-1, 3.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = phi(X, Y, points)

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp)

    ax.set_title('Quality Desenity Function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
