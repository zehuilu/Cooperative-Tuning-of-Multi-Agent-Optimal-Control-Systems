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

    configDict = {"timeStep": 0.10, "timeHorizon": 6.0, "inputBounds": [[0.0, -2E19], [2E19, 2E19]]}
    # configDict = {"timeStep": 0.10, "timeHorizon": 6.0}

    # define a graph with 5 agents by an adjacency matrix
    # row i is the adjacency for agent i

    # for periodic graph
    adjacencyMat1 = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1]])
    adjacencyMat2 = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 1]])
    adjacencyMat3 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 0, 0, 1]])
    adjacencyMatList = [adjacencyMat1, adjacencyMat2, adjacencyMat3]
    # initialize a list of agents which are optimal control systems
    listOcSystem = list()
    for idx in range(adjacencyMatList[0].shape[0]):
        listOcSystem.append(OcSystem(DynSystem=Unicycle(configDict), configDict=configDict))
    # initialize multiple agents in a Cooperative Tuning Framework
    MyMultiPDP = MultiPDP(listOcSystem, adjacencyMatList, graphPeriodicFlag=True)


    # # for static graph
    # adjacencyMat = np.array([
    #     [1, 1, 0, 0],
    #     [1, 1, 1, 1],
    #     [0, 1, 1, 1],
    #     [0, 1, 1, 1]])
    # # initialize a list of agents which are optimal control systems
    # listOcSystem = list()
    # for idx in range(adjacencyMat.shape[0]):
    #     listOcSystem.append(OcSystem(DynSystem=Unicycle(configDict), configDict=configDict))
    # # initialize multiple agents in a Cooperative Tuning Framework
    # MyMultiPDP = MultiPDP(listOcSystem, adjacencyMat)


    # initial state and theta
    initialThetaAll = MyMultiPDP.generateRandomInitialTheta(radius=3, center=[0, 0], headingRange=[0.0])
    initialStateAll = MyMultiPDP.generateRandomInitialState(initialThetaAll, radius=10)

    print("initialStateAll:")
    print(initialStateAll)
    print("initialThetaAll:")
    print(initialThetaAll)

    paraDict = {"stepSize": 0.1, "maxIter": 100}

    # run the algorithm
    MyMultiPDP.solve(initialStateAll, initialThetaAll, paraDict=paraDict)
