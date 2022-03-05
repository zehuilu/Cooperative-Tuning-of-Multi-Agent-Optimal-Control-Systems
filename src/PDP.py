#!/usr/bin/env python3
import casadi
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp


class PDP:
    optMethodStr: str  # a string for optimization method
    muMomentum: float  # for Nesterov Accelerated Gradient method
    velocityNesterov: np.array  # 1d numpy array, velocity for Nesterov
    matI: np.array  # 2d numpy array, identity matrix

    def __init__(self, OcSystem):
        self.OcSystem = OcSystem
        self.DynSystem = OcSystem.DynSystem
        self.configDict = OcSystem.configDict

        self.xXi = casadi.SX.sym("xXi", self.DynSystem.dimStatesAll)
        self.uXi = casadi.SX.sym("uXi", self.DynSystem.dimInputsAll)
        self.xi = casadi.vertcat(self.xXi, self.uXi)
        self.matI = np.eye(self.DynSystem.dimStates)  # identity matrix
        self.diffPMP()

    def diffPMP(self):
        # define the Hamiltonian function
        self.costates = casadi.SX.sym("lambda", self.DynSystem.dimStates)
        self.theta = self.DynSystem.theta
        # stage Hamiltonian
        self.stageHamilton = self.DynSystem._stageCostFun(self.DynSystem.states, self.DynSystem.inputs, self.theta) + \
                             casadi.dot(self.DynSystem.discDynFun(self.DynSystem.states, self.DynSystem.inputs), self.costates)
        # terminal Hamiltonian
        self.terminalHamilton = self.DynSystem._terminalCostFun(self.DynSystem.states, self.theta)

        # loss function
        self.loss = self.DynSystem._lossFun(self.xXi, self.uXi, self.theta)
        self.dLdXi = casadi.jacobian(self.loss, self.xi)
        self.dLdTheta = casadi.jacobian(self.loss, self.theta) # partial derivative
        self.lossFun = casadi.Function("lossFun", [self.xi, self.theta], [self.loss])
        self.dLdXiFun = casadi.Function("dLdXiFun", [self.xi, self.theta], [self.dLdXi])
        # this is a function of partial derivative
        self.dLdThetaFun = casadi.Function("dLdThetaFun", [self.xi, self.theta], [self.dLdTheta])

        # differentiating the dynamics
        self.dfdx = casadi.jacobian(self.DynSystem.discDyn, self.DynSystem.states)
        self.dfdu = casadi.jacobian(self.DynSystem.discDyn, self.DynSystem.inputs)
        # de means derivative w.r.t. theta
        self.dfde = casadi.jacobian(self.DynSystem.discDyn, self.DynSystem.theta)

        self.dfdxFun = casadi.Function("dfdxFun", [self.DynSystem.states, self.DynSystem.inputs, self.theta], [self.dfdx])
        self.dfduFun = casadi.Function("dfduFun", [self.DynSystem.states, self.DynSystem.inputs, self.theta], [self.dfdu])
        self.dfdeFun = casadi.Function("dfdeFun", [self.DynSystem.states, self.DynSystem.inputs, self.theta], [self.dfde])

        # first-order derivatives of stage Hamiltonian, row vector
        self.dHdx = casadi.jacobian(self.stageHamilton, self.DynSystem.states).T
        self.dHdu = casadi.jacobian(self.stageHamilton, self.DynSystem.inputs).T

        self.dHdxFun = casadi.Function('dHdxFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.theta], [self.dHdx])
        self.dHduFun = casadi.Function('dHduFun', [self.DynSystem.states, self.DynSystem.inputs,
            self.costates, self.theta], [self.dHdu])

        # second-order derivatives of stage Hamiltonian
        self.ddHdxdx = casadi.jacobian(self.dHdx, self.DynSystem.states)
        self.ddHdxdu = casadi.jacobian(self.dHdx, self.DynSystem.inputs)
        self.ddHdxde = casadi.jacobian(self.dHdx, self.theta)
        self.ddHdudx = casadi.jacobian(self.dHdu, self.DynSystem.states)
        self.ddHdudu = casadi.jacobian(self.dHdu, self.DynSystem.inputs)
        self.ddHdude = casadi.jacobian(self.dHdu, self.theta)

        self.ddHdxdxFun = casadi.Function('ddHdxdxFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.theta], [self.ddHdxdx])
        self.ddHdxduFun = casadi.Function('ddHdxduFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.theta], [self.ddHdxdu])
        self.ddHdxdeFun = casadi.Function('ddHdxdeFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.theta], [self.ddHdxde])
        self.ddHdudxFun = casadi.Function('ddHdudxFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.theta], [self.ddHdudx])
        self.ddHduduFun = casadi.Function('ddHduduFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.theta], [self.ddHdudu])
        self.ddHdudeFun = casadi.Function('ddHdudeFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.theta], [self.ddHdude])

        # first-order derivatives of terminal Hamiltonian, row vector
        self.dhdx = casadi.jacobian(self.terminalHamilton, self.DynSystem.states).T
        self.dhdxFun = casadi.Function('dhdxFun', [self.DynSystem.states, self.theta], [self.dhdx])

        # second-order derivatives of terminal Hamiltonian
        self.ddhdxdx = casadi.jacobian(self.dhdx, self.DynSystem.states)
        self.ddhdxde = casadi.jacobian(self.dhdx, self.theta)

        self.ddhdxdxFun = casadi.Function('ddhdxdxFun', [self.DynSystem.states, self.theta], [self.ddhdxdx])
        self.ddhdxdeFun = casadi.Function('ddhdxdeFun', [self.DynSystem.states, self.theta], [self.ddhdxde])

    def solve(self, initialState, thetaInit, paraDict: dict):
        # load the optimization function based on paraDict
        self.loadOptFunction(paraDict)

        # initialize some variables for optimization methods
        if (self.optMethodStr == "Vanilla"):
            pass
        elif (self.optMethodStr == "Nesterov"):
            # initialization for Nesterov
            self.velocityNesterov = np.zeros(self.DynSystem.dimParameters)
        else:
            raise Exception("Wrong optimization method type!")

        # initialize the problem and visualize it later
        resultDict = self.OcSystem.solve(initialState, thetaInit)
        self.DynSystem.visualize(resultDict, initialState, thetaInit, blockFlag=False)

        lossTraj = list()
        thetaTraj = list()
        thetaNow = thetaInit
        for idx in range(int(paraDict["maxIter"])):
            # update theta
            lossNow, thetaNext, _ = self.optFun(initialState, thetaNow, paraDict)

            #if lossNow > 1E3:
                #break

            lossTraj.append(lossNow)
            thetaTraj.append(thetaNow)
            thetaNow = thetaNext

            # if idx % 50 == 0:
                # print('Iter:', idx, ' loss:', lossNow)

            print('Iter:', idx, ' loss:', lossNow)

        resultDict = self.OcSystem.solve(initialState, thetaNow)
        lossNow = self.lossFun(resultDict["xi"], thetaNow).full()[0, 0]

        lossTraj.append(lossNow)
        thetaTraj.append(thetaNow)
        print('Last one', ' loss:', lossNow)
        print("Theta: ", thetaNow)

        # visualize
        self.DynSystem.visualize(resultDict, initialState, thetaNow, blockFlag=False)
        # plot the loss
        self.plotLossTraj(lossTraj, blockFlag=False)
        plt.show()

    def loadOptFunction(self, paraDict: dict):
        """
        Load the optimization function. Now support Vanilla gradient descent, Nesterov Momentum.
        Input:
            paraDict: a dictionary which includes the parameters.
        
        Usage:
            # This is for Vanilla gradient descent
            paraDict = {"stepSize": 0.01, "maxIter": 1000, "method": "Vanilla"}

            # This is for Nesterov Momentum
            paraDict = {"stepSize": 0.01, "maxIter": 1000, "method": "Nesterov", "mu": 0.9, "realLossFlag": False}

            self.loadOptFunction(paraDict)
        """
        # the optimization method
        self.optMethodStr = paraDict["method"]

        if (self.optMethodStr == "Vanilla"):
            self.optFun = lambda initialState, thetaNow, paraDict: \
                self.gradientDescentVanilla(initialState, thetaNow, paraDict)
        elif (self.optMethodStr == "Nesterov"):
            self.muMomentum = paraDict["mu"]
            self.optFun = lambda initialState, thetaNow, paraDict: \
                self.Nesterov(initialState, thetaNow, paraDict)
        else:
            raise Exception("Wrong optimization method type!")

    def computeGradient(self, initialState, thetaNow):
        resultDict = self.OcSystem.solve(initialState, thetaNow)
        lqrSystem = self.getLqrSystem(resultDict, thetaNow)
        resultLqr = self.solveLqr(lqrSystem)

        lossNow = self.lossFun(resultDict["xi"], thetaNow).full()[0, 0]

        dLdXi = self.dLdXiFun(resultDict["xi"], thetaNow)
        dXidTheta = np.vstack((np.concatenate(resultLqr["XTrajList"], axis=0),
            np.concatenate(resultLqr["UTrajList"], axis=0)))
        # this is partial derivative
        dLdTheta = self.dLdThetaFun(resultDict["xi"], thetaNow)
        # this is full derivative
        gradient = np.array(np.dot(dLdXi, dXidTheta) + dLdTheta).flatten()

        return lossNow, gradient

    def gradientDescentVanilla(self, initialState, thetaNow, paraDict: dict):
        """
        Vanilla gradient descent method.
        """
        lossNow, gradient = self.computeGradient(initialState, thetaNow)
        thetaNext = thetaNow - paraDict["stepSize"] * gradient
        return lossNow, thetaNext, thetaNow

    def Nesterov(self, initialState, thetaNow, paraDict: dict):
        """
        Nesterov Accelerated Gradient method (NAG).
        """
        # compute the lookahead parameter
        thetaMomentum = thetaNow + self.muMomentum * self.velocityNesterov
        # compute the loss and gradient
        lossNow, gradient = self.computeGradient(initialState, thetaMomentum)
        # update velocity vector for Nesterov
        self.velocityNesterov = self.muMomentum * self.velocityNesterov - \
            paraDict["stepSize"] * np.array(gradient)
        # update the parameter
        thetaNext = thetaNow + self.velocityNesterov

        if paraDict["realLossFlag"]:
            # compute the loss and gradient
            lossNow, _ = self.computeGradient(initialState, thetaNow)
        return lossNow, thetaNext, thetaNow



    def test_func(self, idx):
        xNow = self.xTraj[idx, :]
        uNow = self.uTraj[idx, :]
        lambdaNext = self.costateTraj[idx, :]  # costate

        dynF = np.array(self.dfdxFun(xNow, uNow, self.theta).full())
        dynG = np.array(self.dfduFun(xNow, uNow, self.theta).full())
        dynE = np.array(self.dfdeFun(xNow, uNow, self.theta).full())

        Hxx = np.array(self.ddHdxdxFun(xNow, uNow, lambdaNext, self.theta).full())
        Hxu = np.array(self.ddHdxduFun(xNow, uNow, lambdaNext, self.theta).full())
        Hux = np.array(self.ddHdudxFun(xNow, uNow, lambdaNext, self.theta).full())
        Huu = np.array(self.ddHduduFun(xNow, uNow, lambdaNext, self.theta).full())

        Hxe = np.array(self.ddHdxdeFun(xNow, uNow, lambdaNext, self.theta).full())
        Hue = np.array(self.ddHdudeFun(xNow, uNow, lambdaNext, self.theta).full())
        return dynF, dynG, dynE, Hxx, Hxu, Hux, Huu, Hxe, Hue

    def getLqrSystem_test(self, resultDict: dict, theta):
        """
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

        costateTraj: 2d numpy array, each row is the costate at a time step
        """
        # define functions
        self.xTraj = resultDict["xTraj"]
        self.uTraj = resultDict["uTraj"]
        self.costateTraj = resultDict["costateTraj"]
        self.theta = theta

        # initialize the system matrices of the auxiliary LQR control system
        dynF, dynG, dynE = list(), list(), list()
        Hxx, Hxu, Hxe, Hux, Huu, Hue, hxx, hxe = list(), list(), list(), list(), list(), list(), list(), list()
        # Hxe, dimStates by dimParameters
        # Hue, dimInputs by dimParameters
        # hxe, dimStates by dimParameters

        t0 = time.time()

        processPool = mp.Pool()
        results = processPool.map(self.test_func, np.arange(self.uTraj.shape[0]))

        print("results[0]: ", results[0])
        t1 = time.time()
        print("This loop time used [sec]: ", t1 - t0)



    def getLqrSystem(self, resultDict: dict, theta):
        """
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

        costateTraj: 2d numpy array, each row is the costate at a time step
        """
        # define functions
        xTraj = resultDict["xTraj"]
        uTraj = resultDict["uTraj"]
        costateTraj = resultDict["costateTraj"]

        # initialize the system matrices of the auxiliary LQR control system
        dynF, dynG, dynE = list(), list(), list()
        Hxx, Hxu, Hxe, Hux, Huu, Hue, hxx, hxe = list(), list(), list(), list(), list(), list(), list(), list()
        # Hxe, dimStates by dimParameters
        # Hue, dimInputs by dimParameters
        # hxe, dimStates by dimParameters

        # t0 = time.time()
        # compute the matrices, skip the initial condition and terminal state
        for idx in range(uTraj.shape[0]):
            xNow = xTraj[idx, :]
            uNow = uTraj[idx, :]
            lambdaNext = costateTraj[idx, :]  # costate

            dynF.append(np.array(self.dfdxFun(xNow, uNow, theta).full()))
            dynG.append(np.array(self.dfduFun(xNow, uNow, theta).full()))
            dynE.append(np.array(self.dfdeFun(xNow, uNow, theta).full()))

            Hxx.append(np.array(self.ddHdxdxFun(xNow, uNow, lambdaNext, theta).full()))
            Hxu.append(np.array(self.ddHdxduFun(xNow, uNow, lambdaNext, theta).full()))
            Hux.append(np.array(self.ddHdudxFun(xNow, uNow, lambdaNext, theta).full()))
            Huu.append(np.array(self.ddHduduFun(xNow, uNow, lambdaNext, theta).full()))

            Hxe.append(np.array(self.ddHdxdeFun(xNow, uNow, lambdaNext, theta).full()))
            Hue.append(np.array(self.ddHdudeFun(xNow, uNow, lambdaNext, theta).full()))
        # t1 = time.time()
        # print("This loop time used [sec]: ", t1 - t0)

        hxx.append(np.array(self.ddhdxdxFun(xTraj[-1, :], theta).full()))
        hxe.append(np.array(self.ddhdxdeFun(xTraj[-1, :], theta).full()))

        lqrSystem = {"dynF": dynF,
                     "dynG": dynG,
                     "dynE": dynE,
                     "Hxx": Hxx,
                     "Hxu": Hxu,
                     "Hxe": Hxe,
                     "Hux": Hux,
                     "Huu": Huu,
                     "Hue": Hue,
                     "hxx": hxx,
                     "hxe": hxe}
        return lqrSystem

    def solveLqr(self, lqrSystem):
        # solve the Riccati equations
        PList = self.DynSystem.horizonSteps * [np.zeros((self.DynSystem.dimStates, self.DynSystem.dimStates))]
        WList = self.DynSystem.horizonSteps * [np.zeros((self.DynSystem.dimStates, self.DynSystem.dimParameters))]

        PList[-1] = lqrSystem["hxx"][0]
        WList[-1] = lqrSystem["hxe"][0]
        for idx in range(self.DynSystem.horizonSteps - 1, 0, -1):
            PNext = PList[idx]
            WNext = WList[idx]
            invHuu = np.linalg.inv(lqrSystem["Huu"][idx])
            GinvHuu = np.matmul(lqrSystem["dynG"][idx], invHuu)
            HxuinvHuu = np.matmul(lqrSystem["Hxu"][idx], invHuu)
            At = lqrSystem["dynF"][idx] - np.matmul(GinvHuu, np.transpose(lqrSystem["Hxu"][idx]))
            Rt = np.matmul(GinvHuu, np.transpose(lqrSystem["dynG"][idx]))
            Mt = lqrSystem["dynE"][idx] - np.matmul(GinvHuu, lqrSystem["Hue"][idx])
            Qt = lqrSystem["Hxx"][idx] - np.matmul(HxuinvHuu, np.transpose(lqrSystem["Hxu"][idx]))
            Nt = lqrSystem["Hxe"][idx] - np.matmul(HxuinvHuu, lqrSystem["Hue"][idx])

            tempMat = np.matmul(np.transpose(At), np.linalg.inv(self.matI + np.matmul(PNext, Rt)))
            PNow = Qt + np.matmul(tempMat, np.matmul(PNext, At))
            WNow = Nt + np.matmul(tempMat, WNext + np.matmul(PNext, Mt))

            PList[idx - 1] = PNow
            WList[idx - 1] = WNow

        # compute the trajectory using the Raccti matrices obtained from the above
        # initial condition is zeros
        XTrajList = (self.DynSystem.horizonSteps + 1) * [np.zeros((self.DynSystem.dimStates, self.DynSystem.dimParameters))]
        UTrajList = self.DynSystem.horizonSteps * [np.zeros((self.DynSystem.dimInputs, self.DynSystem.dimParameters))]
        for idx in range(self.DynSystem.horizonSteps):
            PNext = PList[idx]
            WNext = WList[idx]
            invHuu = np.linalg.inv(lqrSystem["Huu"][idx])
            GinvHuu = np.matmul(lqrSystem["dynG"][idx], invHuu)
            At = lqrSystem["dynF"][idx] - np.matmul(GinvHuu, np.transpose(lqrSystem["Hxu"][idx]))
            Mt = lqrSystem["dynE"][idx] - np.matmul(GinvHuu, lqrSystem["Hue"][idx])
            Rt = np.matmul(GinvHuu, np.transpose(lqrSystem["dynG"][idx]))

            XNow = XTrajList[idx]
            UNow = -np.matmul(invHuu, np.matmul(np.transpose(lqrSystem["Hxu"][idx]), XNow) + lqrSystem["Hue"][idx]) \
                   - np.linalg.multi_dot([invHuu, np.transpose(lqrSystem["dynG"][idx]), np.linalg.inv(self.matI + np.dot(PNext, Rt)),
                   (np.matmul(np.matmul(PNext, At), XNow) + np.matmul(PNext, Mt) + WNext)])

            xNext = np.matmul(lqrSystem["dynF"][idx], XNow) + np.matmul(lqrSystem["dynG"][idx], UNow) + lqrSystem["dynE"][idx]
            XTrajList[idx + 1] = xNext
            UTrajList[idx] = UNow

        result = {"XTrajList": XTrajList,
                  "UTrajList": UTrajList}
        return result

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
