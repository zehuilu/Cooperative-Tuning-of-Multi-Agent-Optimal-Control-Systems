#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd() + '/src')
from Unicycle import Unicycle
import time
import numpy as np


if __name__ == '__main__':
    configDict = {"timeStep": 0.1, "timeHorizon": 0.3}
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

    # xAll = np.zeros(MyUnicycle.dimStatesAll)
    # uAll = np.zeros(MyUnicycle.dimInputsAll)
    # uAll[-1] = 5
    # uAll[0] = 3

    # xAll = np.arange(0.0, MyUnicycle.dimStatesAll, dtype=np.float32)
    # uAll = np.arange(0.0, MyUnicycle.dimInputsAll, dtype=np.float32)
    # theta = np.ones(MyUnicycle.dimParameters)

    xAll = np.random.uniform(0, 1, MyUnicycle.dimStatesAll)
    uAll = np.random.uniform(0, 1, MyUnicycle.dimInputsAll)
    theta = np.random.uniform(0, 1, MyUnicycle.dimParameters)
    
    xDecision = np.concatenate((xAll, uAll))

    # x0 = np.array([1., 2., -1.])
    # xGoal = np.array([1., -1., 2.])
    x0 = np.random.uniform(0, 1, MyUnicycle.dimStates)
    xGoal = np.random.uniform(0, 1, MyUnicycle.dimStates)

    print("Cost function for optimal control:")
    t0 = time.time()
    print(MyUnicycle.costFun(xAll, uAll, x0, xGoal, theta))
    t1 = time.time()
    print("cost function time [sec]: ", t1 - t0)

    MyUnicycle.testDynamicsConstraints(x0)
