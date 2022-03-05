#!/usr/bin/env python3
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from PDP import PDP


class MultiPDP:
    numAgent: int  # number of agents
    optMethodStr: str  # a string for optimization method

    def __init__(self, listOcSystem: list, adjacencyMat: np.array):
        self.listOcSystem = listOcSystem
        self.numAgent = len(listOcSystem)
        self.configDict = listOcSystem[0].configDict
        self.adjacencyMat = adjacencyMat
        self.generateMetropolisWeight()
        self.listPDP = list()
        for idx in range(self.numAgent):
            self.listPDP.append(PDP(OcSystem=self.listOcSystem[idx]))

    def generateMetropolisWeight(self):
        """
        Generate a Metropolis Weight matrix, whose entry in i-th row and j-th column is the weight for receiver i and sender j
        """
        # self.weightMat[i][j] is the weight for receiver i and sender j
        self.weightMat = np.zeros((self.numAgent, self.numAgent))
        # each row with index i is a vector of weights given this receiver i

        # for self.adjacencyMat, row i is the adjacency for agent i
        # i-th element in dArray is d_i, namely the number of neighbors (incuding itself)
        dArray = np.sum(self.adjacencyMat, axis=1)

        for row in range(self.numAgent):
            for col in range(self.numAgent):
                # not compute self.weightMat[row][col] yet
                if row != col:
                    # if col (j) is a neighbor of row (i), do calculation; otherwise 0
                    if self.adjacencyMat[row][col] > 0.5:
                        self.weightMat[row][col] = 1 / (max(dArray[row], dArray[col]))

        # sum all the non-diagonal weights, then distracted by a vector with ones
        weightMatSelf = np.ones((self.numAgent)) - np.sum(self.weightMat, axis=1)
        # allocate weights for diagonal
        for idx in range(self.numAgent):
            self.weightMat[idx][idx] = weightMatSelf[idx]

    def generateRandomInitialState(self, radius: float, numAgent: int, headingRange=[-3.14, 3.14]):
        """
        Randomly generate initial state for multiple agents, where the position is randomly distributed on a circle with given radius.

        Inputs:
            radius: the radius of the circle
            numAgent: number of agents
            headingRange: 1d list, the random range of heading angle, [lower bound, upper bound]; to be a deterministic value when the list size is just 1
        
        Outputs:
            initialStateAll: 2d numpy array, i-th row is the initial state for agent-i
        """
        initialStateAll = np.zeros((numAgent, self.listOcSystem[0].DynSystem.dimStates))
        for idx in range(numAgent):
            angle = random.uniform(-3.14, 3.14)
            px = radius * round(math.cos(angle), 2)
            py = radius * round(math.sin(angle), 2)
            if len(headingRange) > 1:
                heading = round(random.uniform(headingRange[0], headingRange[1]), 2)
            else:
                heading = headingRange[0]
            initialStateAll[idx, :] = np.array([px, py, heading])
        return initialStateAll

    def solve(self, initialStateAll, thetaInitAll, paraDict: dict):
        """
        
        Inputs:
            initialStateAll: 2d numpy array, i-th row is an initial state for agent i
            thetaInitAll: 2d numpy array, i-th row is an initial theta for agent i
        """
        # initialize the problem
        resultDictAll = list()
        for idx in range(self.numAgent):
            resultDictAll.append(self.listOcSystem[idx].solve(initialStateAll[idx], thetaInitAll[idx]))

        thetaNowAll = thetaInitAll
        lossTraj = list()
        thetaAllTraj  = list()
        for idxIter in range(int(paraDict["maxIter"])):
            # compute the gradients
            lossNow, lossVecNow, gradientMatNow = self.computeGradient(initialStateAll, thetaNowAll)
            # exchange information and update theta
            thetaNextAll = np.matmul(self.weightMat, thetaNowAll) - paraDict["stepSize"] * gradientMatNow

            lossTraj.append(lossNow)
            thetaAllTraj.append(thetaNowAll)

            thetaNowAll = thetaNextAll

            # if idx % 50 == 0:
                # print('idx:', idxIter, ' loss:', lossNow)

            print('Iter:', idxIter, ' loss:', lossNow)

        # last one
        resultDictList = list()
        lossVec = np.zeros((self.numAgent))
        for idx in range(self.numAgent):
            resultDictList.append(self.listOcSystem[idx].solve(initialStateAll[idx], thetaNowAll[idx]))
            lossVec[idx] = self.listPDP[idx].lossFun(resultDictList[idx]["xi"], thetaNowAll[idx]).full()[0, 0]

        lossTraj.append(lossVec.sum())
        thetaAllTraj.append(thetaNowAll)

        print('Last one', ' loss:', lossVec.sum())
        print("Theta: ", thetaAllTraj)

        # # visualize
        # self.DynSystem.visualize(resultDict, initialState, thetaNow, blockFlag=False)

        # plot the loss
        self.plotLossTraj(lossTraj, blockFlag=False)

        plt.show()

    def computeGradient(self, initialStateAll, thetaNowAll):
        lossVec = np.zeros((self.numAgent))
        # i-th row is the full gradient for agent-i
        gradientMat = np.zeros((self.numAgent, self.listOcSystem[0].DynSystem.dimParameters))
        for idx in range(self.numAgent):
            resultDict = self.listOcSystem[idx].solve(initialStateAll[idx], thetaNowAll[idx])
            lqrSystem = self.listPDP[idx].getLqrSystem(resultDict, thetaNowAll[idx])
            resultLqr = self.listPDP[idx].solveLqr(lqrSystem)

            lossVec[idx] = self.listPDP[idx].lossFun(resultDict["xi"], thetaNowAll[idx]).full()[0, 0]
            dLdXi = self.listPDP[idx].dLdXiFun(resultDict["xi"], thetaNowAll[idx])
            dXidTheta = np.vstack((np.concatenate(resultLqr["XTrajList"], axis=0),
                np.concatenate(resultLqr["UTrajList"], axis=0)))
            # this is partial derivative
            dLdTheta = self.listPDP[idx].dLdThetaFun(resultDict["xi"], thetaNowAll[idx])

            # this is full derivative
            gradientMat[idx, :] = np.array(np.dot(dLdXi, dXidTheta) + dLdTheta).flatten()

        return lossVec.sum(), lossVec, gradientMat

    def plotLossTraj(self, lossTraj, blockFlag=True):
        _, ax1 = plt.subplots(1, 1)
        ax1.plot(np.arange(len(lossTraj)), lossTraj, color="blue", linewidth=2)
        ax1.set_title("Loss")
        ax1.legend(["Loss"])
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")

        if blockFlag:
            plt.show()
        else:
            plt.show(block=False)
