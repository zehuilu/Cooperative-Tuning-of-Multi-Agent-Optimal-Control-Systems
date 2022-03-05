#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd() + '/src')
from PDP import PDP
from OcSystem import OcSystem
from Unicycle import Unicycle
import numpy as np


if __name__ == '__main__':
    # "inputBounds" : [ [lb_input_1, lb_input_2, ...], [ub_input_1, ub_input_2, ...] ]
    # configDict = {"timeStep": 0.1, "timeHorizon": 4.0, "inputBounds": [[-2.0, -0.5], [2.0, 0.5]]}
    configDict = {"timeStep": 0.1, "timeHorizon": 4.0}

    MyDynSystem = Unicycle(configDict)
    MyOcSystem = OcSystem(DynSystem=MyDynSystem, configDict=configDict)
    MyPDP = PDP(OcSystem=MyOcSystem)

    # initial state
    x0 = np.array([0.0, 0.0, 0.0])
    theta = np.array([0.8, 0.2, 0.1])

    paraDict = {"stepSize": 0.05, "maxIter": 10, "method": "Vanilla"}
    # paraDict = {"stepSize": 0.10, "maxIter": 500, "method": "Nesterov", "mu": 0.9, "realLossFlag": False}

    MyPDP.solve(x0, theta, paraDict=paraDict)
