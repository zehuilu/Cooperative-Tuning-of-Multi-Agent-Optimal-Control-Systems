#!/usr/bin/env python3
import casadi
import numpy as np
import time
import matplotlib.pyplot as plt


class Unicycle:
    timeStep: float  # length of time step for Euler integration
    timeHorizon: float  # time horizon for optimal control [sec]
    horizonSteps: int  # number of steps over the horizon
    dimStates: int  # dimension of states
    dimInputs: int  # dimension of inputs
    dimParameters: int  # dimension of tunable parameters

    def __init__(self, configDict):
        self.timeStep = configDict["timeStep"]
        self.timeHorizon = configDict["timeHorizon"]
        self.horizonSteps = round(self.timeHorizon / self.timeStep)
        self.dimStates = 3
        self.dimInputs = 2

        # initial condition is included
        self.dimStatesAll = (self.horizonSteps + 1) * self.dimStates
        self.dimInputsAll = self.horizonSteps * self.dimInputs

        # construct the dynamical functions
        self.buildFunctions()

    def buildFunctions(self):
        """
        Construct dynamics functions in continuous-time and discrete-time.
        Continuous-time Dynamics:
            x_dot = f(x,u)
            x = [x1; x2; x3] = [px; py; psi]
            u = [u1; u2] = [uv; uw]
        Discrete-time Dynamics:
            By Euler Integration:
            xNext = xNow + dt * f(xNow, uNow)
        """
        self.states = casadi.SX.sym("x", self.dimStates)
        self.inputs = casadi.SX.sym("u", self.dimInputs)

        self.dimParameters = self.dimStates
        self.theta = casadi.SX.sym("theta", self.dimParameters)

        # continuous-time dynamical function in casadi SX
        self.contDyn = casadi.vertcat(
            self.inputs[0] * casadi.cos(self.states[2]),
            self.inputs[0] * casadi.sin(self.states[2]),
            self.inputs[1])

        # discrete-time dynamical function in casadi SX
        self.discDyn = self.states + self.timeStep * self.contDyn

        # in casadi.Function
        self.contDynFun = casadi.Function("contDynFun", [self.states, self.inputs], [self.contDyn])
        self.discDynFun = casadi.Function("discDynFun", [self.states, self.inputs], [self.discDyn])

        xAll = casadi.SX.sym("xAll", self.dimStatesAll)
        uAll = casadi.SX.sym("uAll", self.dimInputsAll)
        self.xDecision = casadi.vertcat(xAll, uAll)

        self.costFun = casadi.Function("costFun", [xAll, uAll, self.theta], [self._costFun(xAll, uAll, self.theta)])

        self.dynConstraintsFun = casadi.Function("dynConstraints", [xAll, uAll], [self._dynConstraints(xAll, uAll)])

    def _costFun(self, xAll, uAll, theta):
        cost = 0.0
        # the stage cost from t=0 to t=T-1
        for idx in range(self.horizonSteps):
            xNow = xAll[self.dimStates*idx : self.dimStates*(idx+1)]
            uNow = uAll[self.dimInputs*idx : self.dimInputs*(idx+1)]
            cost += self._stageCostFun(xNow, uNow, theta)

        xTerminal = xAll[self.dimStatesAll-self.dimStates:]
        cost += self._terminalCostFun(xTerminal, theta)

        return cost

    def _stageCostFun(self, xNow, uNow, theta):
        cost = 2 * ((xNow[0]-theta[0])**2 + (xNow[1]-theta[1])**2)
        cost += (uNow[0] ** 2 + uNow[1] ** 2)
        # cost += 2 * (1 - casadi.cos(xNow[2]) * casadi.cos(theta[2]) - casadi.sin(xNow[2]) * casadi.sin(theta[2]))
        return cost
    
    def _terminalCostFun(self, xNow, theta):
        # headingError = 1 - casadi.cos(xNow[2]) * casadi.cos(theta[2]) - casadi.sin(xNow[2]) * casadi.sin(theta[2])
        # cost = 5 * ((xNow[0]-theta[0])**2 + (xNow[1]-theta[1])**2) + 10 * headingError
        cost = 5 * ((xNow[0]-theta[0])**2 + (xNow[1]-theta[1])**2)
        return cost

    def _dynConstraints(self, xAll, uAll):
        dynCons = list()
        for idx in range(self.horizonSteps):
            xNow = xAll[self.dimStates*idx : self.dimStates*(idx+1)]
            uNow = uAll[self.dimInputs*idx : self.dimInputs*(idx+1)]
            xNext = xAll[self.dimStates*(idx+1) : self.dimStates*(idx+2)]
            currentCons = xNext - self.discDynFun(xNow, uNow)
            dynCons = casadi.vertcat(dynCons, currentCons)
        return dynCons

    def _lossFun(self, xAll, uAll, theta):
        xTerminal = xAll[self.dimStatesAll-self.dimStates:]

        # headingError = 1 - casadi.cos(xTerminal[2]) * casadi.cos(theta[2]) - casadi.sin(xTerminal[2]) * casadi.sin(theta[2])
        # loss = 1 * ((xTerminal[0]-theta[0]) ** 2 + (xTerminal[1]-theta[1]) ** 2 + 10 * headingError)

        loss = 10 * ((xTerminal[0]-theta[0]) ** 2 + (xTerminal[1]-theta[1]) ** 2)
        return loss

    def visualize(self, resultDict, initialState, theta, blockFlag=True):
        _, ax1 = plt.subplots(1, 1)
        ax1.plot(resultDict["xTraj"][:,0], resultDict["xTraj"][:,1], color="blue", linewidth=2)
        ax1.plot(resultDict["xTrajOpt"][:,0], resultDict["xTrajOpt"][:,1], color="red", linewidth=2, linestyle="dashed")
        ax1.scatter(initialState[0], initialState[1], marker="o", color="blue")
        ax1.scatter(theta[0], theta[1], marker="*", color="red")
        ax1.set_title("Trajectory")
        ax1.legend(["Optimal Trajectory", "Trajectory from nlp solver", "start", "goal"])
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")

        _, (ax21, ax22, ax23) = plt.subplots(3, 1)
        ax21.plot(resultDict["timeTraj"][:-1], resultDict["uTraj"][:,0], color="blue", linewidth=2)
        ax21.legend(["velocity input"])
        ax21.set_xlabel("time [sec]")
        ax21.set_ylabel("velocity [m/s]")

        ax22.plot(resultDict["timeTraj"][:-1], resultDict["uTraj"][:,1], color="blue", linewidth=2)
        ax22.legend(["angular velocity input"])
        ax22.set_xlabel("time [sec]")
        ax22.set_ylabel("angular velocity [rad/s]")

        ax23.plot(resultDict["timeTraj"], resultDict["xTraj"][:,2], color="blue", linewidth=2)
        ax23.scatter(resultDict["timeTraj"][0], initialState[2], marker="o", color="blue")
        ax23.scatter(resultDict["timeTraj"][-1], theta[2], marker="*", color="red")
        ax23.legend(["Optimal Trajectory", "start", "goal"])
        ax23.set_xlabel("time [sec]")
        ax23.set_ylabel("heading [radian]")

        if blockFlag:
            plt.show()
        else:
            plt.show(block=False)

    def testDynamicsConstraints(self, x0):
        """
        Test the dynamics constraints by forward propagating the dynamics
        given an initial condition and a sequence of inputs.

        Input:
            x0: 1d numpy array, the initial state
            uAll: 1d numpy array, the sequence of inputs
                [u_0(0),u_1(0), u_0(1),u_1(1), ..., u_0(T-1),u_1(T-1)]

        Output:
            timeTraj: 1d numpy array, [0, timeStep, 2*timeStep, ..., timeHorizon]

            xTraj: 2d numpy array, each row is the state at a time step
                [[state_0(0), state_1(0), state_2(0), ...],
                [state_0(1), state_1(1), state_2(1), ...],
                ...
                [state_0(T), state_1(T), state_2(T), ...]]

            uTraj: 2d numpy array, each row is the input at a time step
                [[u_0(0), u_1(0), ...],
                [u_0(1), u_1(1), ...],
                ...
                [u_0(T-1), u_1(T-1), ...]]
        """
        # generate random inputs
        # uAll: 1d numpy array, the sequence of inputs,
        # [u_0(0),u_1(0), u_0(1),u_1(1), ..., u_0(T-1),u_1(T-1)]
        uAll = np.random.uniform(0, 1, self.dimInputsAll)
        xAll = np.zeros(self.dimStatesAll)

        xNow = x0
        xAll[0 : self.dimStates] = np.array(x0).flatten()
        for idx in range(self.horizonSteps):
            xNext = self.discDynFun(xNow, uAll[idx*self.dimInputs : (idx+1)*self.dimInputs])
            xAll[(idx+1)*self.dimStates : (idx+2)*self.dimStates] = np.array(xNext).flatten()
            xNow = xNext

        print("equality constraints: ")
        t0 = time.time()
        eqCon = self.dynConstraintsFun(xAll, uAll)
        print(eqCon)
        t1 = time.time()
        print("equality constraints time [sec]: ", t1 - t0)

        normCheck = np.linalg.norm(eqCon)
        print("equality constraints norm (expected to be near zero): ", normCheck)
