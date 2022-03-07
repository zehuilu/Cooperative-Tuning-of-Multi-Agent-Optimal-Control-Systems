#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd() + '/src')
from MultiPDP import MultiPDP
from OcSystem import OcSystem
from Unicycle import Unicycle
import numpy as np


if __name__ == '__main__':
    configDict = {"timeStep": 0.02, "timeHorizon": 4.0}

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
    initialStateAll = MyMultiPDP.generateRandomInitialState(
        radius=20, numAgent=adjacencyMat.shape[0], center=[0, 0], headingRange=[-1.57, 1.57])
    initialThetaAll = MyMultiPDP.generateRandomInitialState(
        radius=5, numAgent=adjacencyMat.shape[0], center=[0, 0], headingRange=[-0.5, 0.5])

    print("initialStateAll:")
    print(initialStateAll)
    print("initialThetaAll:")
    print(initialThetaAll)

    paraDict = {"stepSize": 0.2, "maxIter": 100}

    # run the algorithm
    MyMultiPDP.solve(initialStateAll, initialThetaAll, paraDict=paraDict)
