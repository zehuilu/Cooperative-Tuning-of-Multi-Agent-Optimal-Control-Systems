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

    def generateRandomInitialState(self, radius: float, numAgent: int, center=[0.0, 0.0], headingRange=[-3.14, 3.14]):
        """
        Randomly generate initial state for multiple agents, where the position is randomly distributed on a circle with given radius and center.

        Inputs:
            radius: the radius of the circle
            numAgent: number of agents
            center; 1d lsit, the position of center of the circle, [px0, py0]
            headingRange: 1d list, the random range of heading angle, [lower bound, upper bound]; to be a deterministic value when the list size is just 1
        
        Outputs:
            initialStateAll: 2d numpy array, i-th row is the initial state for agent-i
        """
        initialStateAll = np.zeros((numAgent, self.listOcSystem[0].DynSystem.dimStates))
        for idx in range(numAgent):
            angle = random.uniform(-3.14, 3.14)
            px = center[0] + radius * round(math.cos(angle), 2)
            py = center[1] + radius * round(math.sin(angle), 2)
            if len(headingRange) > 1:
                heading = round(random.uniform(headingRange[0], headingRange[1]), 2)
            else:
                heading = headingRange[0]
            initialStateAll[idx, :] = np.array([px, py, heading])
        return initialStateAll

    def solve(self, initialStateAll, initialThetaAll, paraDict: dict):
        """
        
        Inputs:
            initialStateAll: 2d numpy array, i-th row is an initial state for agent i
            initialThetaAll: 2d numpy array, i-th row is an initial theta for agent i
        """
        # initialize the problem
        resultDictList = list()
        for idx in range(self.numAgent):
            resultDictList.append(self.listOcSystem[idx].solve(initialStateAll[idx], initialThetaAll[idx]))

        # visualize the initial result
        self.visualize(resultDictList, initialStateAll, initialThetaAll, blockFlag=False, legendFlag=False)

        thetaNowAll = initialThetaAll
        lossTraj = list()
        thetaAllTraj  = list()
        thetaErrorTraj = list()
        idxIterMargin = 40
        for idxIter in range(int(paraDict["maxIter"])):
            # error among theta
            thetaErrorTraj.append(self.computeThetaError(thetaNowAll))
            # compute the gradients
            lossNow, lossVecNow, gradientMatNow = self.computeGradient(initialStateAll, thetaNowAll)
            # exchange information and update theta
            if idxIter < idxIterMargin:
                thetaNextAll = np.matmul(self.weightMat, thetaNowAll) - paraDict["stepSize"] * gradientMatNow
            else:
                thetaNextAll = np.matmul(self.weightMat, thetaNowAll)

            lossTraj.append(lossNow)
            thetaAllTraj.append(thetaNowAll)
            gradientNorm = np.linalg.norm(gradientMatNow, axis=1).sum()
            thetaNowAll = thetaNextAll
            if idxIter >= idxIterMargin:
                gradientNorm = 0.0

            printStr = 'Iter:' + str(idxIter) + ', loss:' + str(lossNow) + ', grad norm:' + str(gradientNorm) + ', theta error:' + str(thetaErrorTraj[idxIter])
            print(printStr)

            if (gradientNorm <= 0.01) and (thetaErrorTraj[idxIter] <= 0.001):
                break

        # last one
        resultDictList = list()
        lossVec = np.zeros((self.numAgent))
        for idx in range(self.numAgent):
            resultDictList.append(self.listOcSystem[idx].solve(initialStateAll[idx], thetaNowAll[idx]))
            lossVec[idx] = self.listPDP[idx].lossFun(resultDictList[idx]["xi"], thetaNowAll[idx]).full()[0, 0]

        # lossTraj.append(lossVec.sum())
        # thetaAllTraj.append(thetaNowAll)
        # thetaErrorTraj.append(self.computeThetaError(thetaNowAll))

        print('Iter:', idxIter + 1, ' loss:', lossVec.sum())

        # plot the loss
        self.plotLossTraj(lossTraj, thetaErrorTraj, blockFlag=False)

        # visualize
        self.visualize(resultDictList, initialStateAll, thetaNowAll, legendFlag=False)

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

    def computeThetaError(self, thetaNowAll):
        error = 0.0
        for i in range(self.numAgent):
            for j in range(self.numAgent):
                error += np.linalg.norm(thetaNowAll[i, :] - thetaNowAll[j, :]) ** 2
        return error

    def plotLossTraj(self, lossTraj, thetaErrorTraj, blockFlag=True):
        _, (ax1, ax2) = plt.subplots(2, 1)
        lossTraj = lossTraj / lossTraj[0]
        ax1.plot(np.arange(len(lossTraj), dtype=int), lossTraj, color="blue", linewidth=2)
        # ax1.set_title("Loss")
        # ax1.legend(["Loss"])
        # ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss (Relative)")

        ax2.plot(np.arange(len(thetaErrorTraj), dtype=int), thetaErrorTraj, color="blue", linewidth=2)
        # ax2.set_title("Theta error")
        # ax2.legend(["error"])
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Error")

        if blockFlag:
            plt.show()
        else:
            plt.show(block=False)

    def plotArrow(self, stateNow):
        magnitude = 0.1
        dx = magnitude * math.cos(stateNow[2])
        dy = magnitude * math.sin(stateNow[2])
        width = 0.03
        plt.arrow(stateNow[0], stateNow[1], dx, dy, width=width, head_width=7*width, head_length=3*width, alpha=0.5, color="green")

    def visualize(self, resultDictList, initialStateAll, thetaAll, blockFlag=True, legendFlag=True):
        _, ax1 = plt.subplots(1, 1)

        for idx in range(self.numAgent):
            ax1.plot(resultDictList[idx]["xTraj"][:,0], resultDictList[idx]["xTraj"][:,1], color="blue", linewidth=2)
            ax1.scatter(initialStateAll[idx, 0], initialStateAll[idx, 1], marker="o", color="blue")
            ax1.scatter(thetaAll[idx, 0], thetaAll[idx, 1], marker="*", color="red")

        # plot arrows for heading angles
        # for idx in range(self.numAgent):
        #     self.plotArrow(initialStateAll[idx, :])
        #     self.plotArrow(resultDictList[idx]["xTraj"][-1, :])

        # ax1.set_title("Trajectory")
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")
        ax1.axis("equal")
        # plot legends
        if legendFlag:
            labels = ["Start", "Goal"]
            marker = ["o", "*"]
            colors = ["blue", "red"]
            f = lambda m,c: plt.plot([], [], marker=m, color=c, ls="none")[0]
            handles = [f(marker[i], colors[i]) for i in range(len(labels))]
            handles.append(plt.plot([],[], linestyle=None, color="blue", linewidth=2)[0])
            labels.append("Trajectory")
            plt.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left", framealpha=1)

        _, (ax21, ax22, ax23) = plt.subplots(3, 1)
        for idx in range(self.numAgent):        
            ax21.plot(resultDictList[idx]["timeTraj"][:-1], resultDictList[idx]["uTraj"][:,0], color="blue", linewidth=2)
            # ax21.legend(["velocity input"])
            ax21.set_xlabel("time [sec]")
            ax21.set_ylabel("velocity [m/s]")

            ax22.plot(resultDictList[idx]["timeTraj"][:-1], resultDictList[idx]["uTraj"][:,1], color="blue", linewidth=2)
            # ax22.legend(["angular velocity input"])s
            ax22.set_xlabel("time [sec]")
            ax22.set_ylabel("angular velocity [rad/s]")

            ax23.plot(resultDictList[idx]["timeTraj"], resultDictList[idx]["xTraj"][:,2], color="blue", linewidth=2)
            ax23.scatter(resultDictList[idx]["timeTraj"][0], initialStateAll[idx, 2], marker="o", color="blue")
            # ax23.scatter(resultDictList[idx]["timeTraj"][-1], thetaAll[idx, 2], marker="*", color="red")
            # ax23.legend(["Optimal Trajectory", "start", "goal"])
            ax23.set_xlabel("time [sec]")
            ax23.set_ylabel("heading [radian]")

        if blockFlag:
            plt.show()
        else:
            plt.show(block=False)
