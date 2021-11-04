#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd() + '/src')
from Unicycle import Unicycle
from OcSystem import OcSystem
import time
import numpy as np


if __name__ == '__main__':
    configDict = {"timeStep": 0.1, "timeHorizon": 4.0, "inputBounds": [[-1.0, 1.0], [-0.5, 0.5]]}
    MyOcSystem = OcSystem(DynSystem=Unicycle(configDict), configDict=configDict)

    # xAll = np.zeros(MyOcSystem.DynSystem.dimStatesAll)
    # uAll = np.zeros(MyOcSystem.DynSystem.dimInputsAll)
    # uAll = uAll.at[0].set(3)
    # xDecision = np.concatenate((xAll, uAll))

    # initial position and goal position
    xGoal = np.array([1., 1., 0.5])
    x0 = np.array([0.1, -0.05, 0.05])
    # set parameter theta for cost function
    theta = np.ones(MyOcSystem.DynSystem.dimParameters)
    theta[-1] = 10.0  # for terminal cost

    t0 = time.time()
    resultDict = MyOcSystem.solve(x0, xGoal, theta)
    t1 = time.time()
    print("Ipopt time [sec]: ", t1 - t0)

    # t0 = time.time()
    # print("cost function: ", np.array(MyOcSystem.costFun(resultDict["xDecision"])))
    # print("cost function gradient: ", np.array(MyOcSystem.costGradFun(resultDict["xDecision"])))
    # t1 = time.time()
    # print("Time used [sec]: ", t1 - t0)

    # t0 = time.time()
    # print("equality constraints: ", np.array(MyOcSystem.eqConstraintsFun(resultDict["xDecision"])))
    # print("equality constraints gradients: ", np.array(MyOcSystem.eqConstraintsGradFun(resultDict["xDecision"])))
    # t1 = time.time()
    # print("Time used [sec]: ", t1 - t0)

    # visualize the result
    MyOcSystem.DynSystem.visualize(resultDict, x0, xGoal)
