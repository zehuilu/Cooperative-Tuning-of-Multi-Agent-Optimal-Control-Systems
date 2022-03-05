#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd() + '/src')
from MultiPDP import MultiPDP
from OcSystem import OcSystem
from Unicycle import Unicycle
import numpy as np


if __name__ == '__main__':
    # "inputBounds" : [ [lb_input_1, lb_input_2, ...], [ub_input_1, ub_input_2, ...] ]
    # configDict = {"timeStep": 0.1, "timeHorizon": 4.0, "inputBounds": [[-2.0, -0.5], [2.0, 0.5]]}
    configDict = {"timeStep": 0.1, "timeHorizon": 4.0}

    # define a graph with 5 agents by an adjacency matrix
    # row i is the adjacency for agent i

    # adjacencyMat = np.array([
    #     [1, 1, 0, 0, 1],
    #     [1, 1, 1, 0, 1],
    #     [0, 1, 1, 1, 0],
    #     [0, 0, 1, 1, 1],
    #     [1, 1, 0, 1, 1]])

    adjacencyMat = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1]])

    # initialize a list of agents which are optimal control systems
    listOcSystem = list()
    for idx in range(adjacencyMat.shape[0]):
        listOcSystem.append(OcSystem(DynSystem=Unicycle(configDict), configDict=configDict))

    # initialize multiple agents in a Cooperative Tuning Framework
    MyMultiPDP = MultiPDP(listOcSystem, adjacencyMat)

    # initial state and theta
    initialStateAll = MyMultiPDP.generateRandomInitialState(radius=2, numAgent=adjacencyMat.shape[0])
    thetaInitAll = MyMultiPDP.generateRandomInitialState(radius=0.5, numAgent=adjacencyMat.shape[0], headingRange=[0.1])

    print("initialStateAll: ", initialStateAll)
    print("thetaInitAll: ", thetaInitAll)

    paraDict = {"stepSize": 0.05, "maxIter": 40}

    # run the algorithm
    MyMultiPDP.solve(initialStateAll, thetaInitAll, paraDict=paraDict)
