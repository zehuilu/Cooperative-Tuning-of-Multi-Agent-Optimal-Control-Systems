#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd() + '/src')
from PDP import PDP
from Unicycle import Unicycle
import time
import numpy as np


if __name__ == '__main__':
    # configDict = {"timeStep": 0.1, "timeHorizon": 4.0, "inputBounds": [[-1.0, 1.0], [-0.5, 0.5]]}
    configDict = {"timeStep": 0.1, "timeHorizon": 4.0, "inputBounds": [[-2E19, 2E19], [-2E19, 2E19]]}

    MyDynSystem = Unicycle(configDict)
    MyPDP = PDP(DynSystem=MyDynSystem, configDict=configDict)

    # initial position and goal position
    xGoal = np.array([5., 5., 0.5])
    x0 = np.array([0.1, -0.05, 0.05])
    # set parameter theta for cost function
    theta = np.ones(MyPDP.DynSystem.dimParameters)
    # theta[-1] = 5.0  # for terminal cost

    # paraDict = {"stepSize": 0.01, "maxIter": 100, "method": "Vanilla"}
    paraDict = {"stepSize": 0.10, "maxIter": 400, "method": "Nesterov", "mu": 0.9, "realLossFlag": False}

    MyPDP.solve(x0, xGoal, theta, paraDict=paraDict)
