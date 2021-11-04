#!/usr/bin/env python3
import casadi
import cyipopt
import numpy as np
import matplotlib.pyplot as plt


class OcSystem:
    def __init__(self, DynSystem, configDict):
        self.DynSystem = DynSystem
        self.configDict = configDict
        # decision variable x here = [xAll; uAll] in DynSystem
        # x[0 : self.DynSystem.dimStatesAll] = xAll
        # x[self.DynSystemdim.StatesAll:] = uAll

    def solve(self, initialState, desiredState, theta):

        _cost = self.DynSystem.costFun(self.DynSystem.xDecision[0:self.DynSystem.dimStatesAll],
            self.DynSystem.xDecision[self.DynSystem.dimStatesAll:], initialState, desiredState, theta)
        _costGrad = casadi.jacobian(_cost, self.DynSystem.xDecision)

        self.costFun = casadi.Function("costFun", [self.DynSystem.xDecision], [_cost])
        self.costGradFun = casadi.Function("costGradFun", [self.DynSystem.xDecision], [_costGrad])

        _eqConstraints = self.DynSystem.dynConstraintsFun(self.DynSystem.xDecision[0:self.DynSystem.dimStatesAll],
            self.DynSystem.xDecision[self.DynSystem.dimStatesAll:], initialState)
        _eqConstraintsGrad = casadi.jacobian(_eqConstraints, self.DynSystem.xDecision)
        self.eqConstraintsFun = casadi.Function("eqConstraintsFun", [self.DynSystem.xDecision], [_eqConstraints])
        self.eqConstraintsGradFun = casadi.Function("eqConstraintsGradFun", [self.DynSystem.xDecision], [_eqConstraintsGrad])

        # compute the starting point
        x0 = self.computeStartingPoint(initialState)

        # initialize the lower and upper bounds
        lb = [-2E19 for _ in range(self.DynSystem.dimStatesAll)]  # no bounds for states
        ub = [2E19 for _ in range(self.DynSystem.dimStatesAll)]  # no bounds for states
        # input bounds
        for _ in range(self.DynSystem.horizonSteps):
            # for each input signal
            for idx in range(self.DynSystem.dimInputs):
                lb.append(self.configDict["inputBounds"][idx][0])
                ub.append(self.configDict["inputBounds"][idx][1])
                # lb.append(-2E19)
                # ub.append(2E19)

        cl = [0.0 for _ in range(self.DynSystem.horizonSteps*self.DynSystem.dimStates)]
        cu = [0.0 for _ in range(self.DynSystem.horizonSteps*self.DynSystem.dimStates)]

        Problem = NLP(OcSystem=self)
        MyProblem = cyipopt.Problem(
            n=len(x0),
            m=len(cl),
            problem_obj=Problem,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu)

        MyProblem.add_option("print_level", 0)
        MyProblem.add_option("sb", "yes")
        MyProblem.add_option("mu_strategy", "adaptive")
        MyProblem.add_option("hessian_approximation", "limited-memory")
        # MyProblem.add_option("derivative_test", "first-order")
        # MyProblem.add_option("derivative_test_perturbation", 1E-6)

        xDecision, info = MyProblem.solve(x0)

        costateTraj = np.array(info["mult_g"]).reshape((self.DynSystem.horizonSteps, self.DynSystem.dimStates))

        xAll = xDecision[0 : self.DynSystem.dimStatesAll]
        uAll = xDecision[self.DynSystem.dimStatesAll:]
        xTrajOpt = xAll.reshape((-1, self.DynSystem.dimStates))
        timeTraj, xTraj, uTraj = self.forwardPropagate(initialState, uAll)

        xi = np.vstack((xTraj.reshape((-1,1)), uTraj.reshape((-1,1))))

        resultDict = {"timeTraj": timeTraj,
                      "xTraj": xTraj,
                      "uTraj": uTraj,
                      "costateTraj": costateTraj,
                      "xTrajOpt": xTrajOpt,
                      "xDecision": xDecision,
                      "xi": xi}

        return resultDict

    def computeStartingPoint(self, initialState):
        """
        Compute the starting point of the optimization problem by forward propagating the dynamics
        given an initial condition and zeros inputs.

        Input:
            initialState: 1d numpy array, the initial state

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
        uAll = np.zeros(self.DynSystem.dimInputsAll)
        xAll = np.zeros(self.DynSystem.dimStatesAll)

        xNow = initialState
        for idx in range(self.DynSystem.horizonSteps):
            xNext = self.DynSystem.discDynFun(
                xNow, uAll[idx*self.DynSystem.dimInputs : (idx+1)*self.DynSystem.dimInputs])
            xAll[idx*self.DynSystem.dimStates : (idx+1)*self.DynSystem.dimStates] = np.array(xNext).flatten()
            xNow = xNext
        xDecision = np.concatenate((xAll, uAll))

        # norm_check = np.linalg.norm(self.eqConstraintsFun(xDecision))
        # print("starting point equality constraint check (should be zero): ", norm_check)

        return xDecision

    def forwardPropagate(self, initialState, uAll):
        """
        Forward propagate the dynamics given an initial condition and a sequence of inputs.

        Input:
            initialState: 1d numpy array, the initial state
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
        timeTraj = np.zeros(self.DynSystem.horizonSteps+1)
        xTraj = np.zeros((self.DynSystem.horizonSteps+1, self.DynSystem.dimStates))
        uTraj = np.zeros((self.DynSystem.horizonSteps, self.DynSystem.dimInputs))

        xNow = initialState
        timeTraj[0] = 0.0  # starting time [sec]
        xTraj[0, :] = np.array(initialState)  # initial state
        for idx in range(self.DynSystem.horizonSteps):
            xNext = self.DynSystem.discDynFun(
                xNow, uAll[idx*self.DynSystem.dimInputs : (idx+1)*self.DynSystem.dimInputs])
            timeTraj[idx+1] = (idx+1) * self.DynSystem.timeStep  # time [sec]

            # casadi array to 1d numpy array
            xTraj[idx+1, :] = np.array(xNext).reshape((1,-1)).flatten()
            uTraj[idx, :] = np.array(
                uAll[idx*self.DynSystem.dimInputs : (idx+1)*self.DynSystem.dimInputs])  # input
            xNow = xNext
        return timeTraj, xTraj, uTraj


class NLP():
    def __init__(self, OcSystem):
        self.OcSystem = OcSystem

    def objective(self, x):
        return self.OcSystem.costFun(x)

    def gradient(self, x):
        return self.OcSystem.costGradFun(x)

    def constraints(self, x):
        return self.OcSystem.eqConstraintsFun(x)

    def jacobian(self, x):
        return self.OcSystem.eqConstraintsGradFun(x)
