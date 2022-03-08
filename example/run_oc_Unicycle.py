#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd() + '/src')
import time
import numpy as np
from Unicycle import Unicycle
from OcSystem import OcSystem


if __name__ == '__main__':
    # "inputBounds" : [ [lb_input_1, lb_input_2, ...], [ub_input_1, ub_input_2, ...] ]

    # configDict = {"timeStep": 0.1, "timeHorizon": 0.5}
    configDict = {"timeStep": 0.1, "timeHorizon": 6.0, "inputBounds": [[0.0, -2E19], [2E19, 2E19]]}

    MyUnicycle = Unicycle(configDict=configDict)

    t0 = time.time()
    print("Unicycle continous dyn: ")
    print(MyUnicycle.contDynFun( np.array([0., 0., 0.]), [0., 0.] ))
    t1 = time.time()
    print("continous dyn time used [sec]: ", t1 - t0)
    print("Unicycle discrete dyn: ")
    print(MyUnicycle.discDynFun( [0., 0., 0.], [0., 0.] ))
    t2 = time.time()
    print("discrete dyn time used [sec]: ", t2 - t1)

    xAll = np.random.uniform(0, 1, MyUnicycle.dimStatesAll)
    uAll = np.random.uniform(0, 1, MyUnicycle.dimInputsAll)
    theta = np.random.uniform(0, 1, MyUnicycle.dimParameters)
    
    xDecision = np.concatenate((xAll, uAll))

    # x0 = np.array([1., 2., -1.])
    x0 = np.random.uniform(0, 1, MyUnicycle.dimStates)

    print("Cost function for optimal control:")
    t0 = time.time()
    print(MyUnicycle.costFun(xAll, uAll, theta))
    t1 = time.time()
    print("cost function time [sec]: ", t1 - t0)

    MyUnicycle.testDynamicsConstraints(x0)

    # initial state
    theta = np.array([0.2, -0.3, 0.0])
    x0 = MyUnicycle.generateRandomInitialState(theta, radius=2, center=[0.0, 0.0])

    # solve
    MyOcSystem = OcSystem(DynSystem=MyUnicycle, configDict=configDict)
    resultDict = MyOcSystem.solve(x0, theta)

    # visualize
    MyUnicycle.visualize(resultDict, x0, theta)
