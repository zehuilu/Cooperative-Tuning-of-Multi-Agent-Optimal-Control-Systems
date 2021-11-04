#!/usr/bin/env python3
import casadi
import numpy as np
import matplotlib.pyplot as plt
from OcSystem import OcSystem


class PDP:
    def __init__(self, DynSystem, configDict):
        self.DynSystem = DynSystem
        self.OcSystem = OcSystem(DynSystem=DynSystem, configDict=configDict)
        self.configDict = configDict

        self.xXi = casadi.SX.sym("xXi", (self.DynSystem.horizonSteps+1) * self.DynSystem.dimStates)
        self.uXi = casadi.SX.sym("uXi", self.DynSystem.horizonSteps * self.DynSystem.dimInputs)
        self.xi = casadi.vertcat(self.xXi, self.uXi)

        self.diffPMP()

    def diffPMP(self):
        # define the Hamiltonian function
        self.costates = casadi.SX.sym("lambda", self.DynSystem.dimStates)
        self.thetaSingle = casadi.SX.sym("thetaSingle", 1)
        # stage Hamiltonian
        self.stageHamilton = self.DynSystem._stageCostFun(self.DynSystem.states, self.DynSystem.inputs,
            self.thetaSingle, self.DynSystem.xGoal) + \
            casadi.dot(self.DynSystem.discDynFun(self.DynSystem.states, self.DynSystem.inputs), self.costates)
        # terminal Hamiltonian
        self.terminalHamilton = self.DynSystem._terminalCostFun(self.DynSystem.states, self.thetaSingle, self.DynSystem.xGoal)

        # loss function
        self.loss = self.DynSystem._lossFun(self.xXi, self.uXi, self.DynSystem.theta, self.DynSystem.xGoal)
        self.dLdXi = casadi.jacobian(self.loss, self.xi)
        self.dLdTheta = casadi.jacobian(self.loss, self.DynSystem.theta)
        self.lossFun = casadi.Function("lossFun", [self.xi, self.DynSystem.theta, self.DynSystem.xGoal], [self.loss])
        self.dLdXiFun = casadi.Function("dLdXiFun", [self.xi, self.DynSystem.theta, self.DynSystem.xGoal], [self.dLdXi])
        self.dLdThetaFun = casadi.Function("dLdThetaFun", [self.xi, self.DynSystem.theta, self.DynSystem.xGoal], [self.dLdTheta])

        # differentiating the dynamics
        self.dfdx = casadi.jacobian(self.DynSystem.discDyn, self.DynSystem.states)
        self.dfdu = casadi.jacobian(self.DynSystem.discDyn, self.DynSystem.inputs)
        self.dfde = casadi.jacobian(self.DynSystem.discDyn, self.DynSystem.theta)

        self.dfdxFun = casadi.Function("dfdxFun", [self.DynSystem.states, 
            self.DynSystem.inputs, self.DynSystem.theta], [self.dfdx])
        self.dfduFun = casadi.Function("dfduFun", [self.DynSystem.states,
            self.DynSystem.inputs, self.DynSystem.theta], [self.dfdu])
        self.dfdeFun = casadi.Function("dfdeFun", [self.DynSystem.states,
            self.DynSystem.inputs, self.DynSystem.theta], [self.dfde])

        # first-order derivatives of stage Hamiltonian, row vector
        self.dHdx = casadi.jacobian(self.stageHamilton, self.DynSystem.states).T
        self.dHdu = casadi.jacobian(self.stageHamilton, self.DynSystem.inputs).T

        self.dHdxFun = casadi.Function('dHdxFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.thetaSingle, self.DynSystem.xGoal], [self.dHdx])
        self.dHduFun = casadi.Function('dHduFun', [self.DynSystem.states, self.DynSystem.inputs,
            self.costates, self.thetaSingle, self.DynSystem.xGoal], [self.dHdu])

        # second-order derivatives of stage Hamiltonian
        self.ddHdxdx = casadi.jacobian(self.dHdx, self.DynSystem.states)
        self.ddHdxdu = casadi.jacobian(self.dHdx, self.DynSystem.inputs)
        self.ddHdxdeSingle = casadi.jacobian(self.dHdx, self.thetaSingle)
        self.ddHdudx = casadi.jacobian(self.dHdu, self.DynSystem.states)
        self.ddHdudu = casadi.jacobian(self.dHdu, self.DynSystem.inputs)
        self.ddHdudeSingle = casadi.jacobian(self.dHdu, self.thetaSingle)

        self.ddHdxdxFun = casadi.Function('ddHdxdxFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.thetaSingle, self.DynSystem.xGoal], [self.ddHdxdx])
        self.ddHdxduFun = casadi.Function('ddHdxduFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.thetaSingle, self.DynSystem.xGoal], [self.ddHdxdu])
        self.ddHdxdeSingleFun = casadi.Function('ddHdxdeSingleFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.thetaSingle, self.DynSystem.xGoal], [self.ddHdxdeSingle])
        self.ddHdudxFun = casadi.Function('ddHdudxFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.thetaSingle, self.DynSystem.xGoal], [self.ddHdudx])
        self.ddHduduFun = casadi.Function('ddHduduFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.thetaSingle, self.DynSystem.xGoal], [self.ddHdudu])
        self.ddHdudeSingleFun = casadi.Function('ddHdudeSingleFun', [self.DynSystem.states, self.DynSystem.inputs, 
            self.costates, self.thetaSingle, self.DynSystem.xGoal], [self.ddHdudeSingle])

        # first-order derivatives of terminal Hamiltonian, row vector
        self.dhdx = casadi.jacobian(self.terminalHamilton, self.DynSystem.states).T
        self.dhdxFun = casadi.Function('dhdxFun', [self.DynSystem.states, 
            self.thetaSingle, self.DynSystem.xGoal], [self.dhdx])

        # second-order derivatives of stage Hamiltonian
        self.ddhdxdx = casadi.jacobian(self.dhdx, self.DynSystem.states)
        self.ddhdxdeSingle = casadi.jacobian(self.dhdx, self.thetaSingle)

        self.ddhdxdxFun = casadi.Function('ddhdxdxFun', [self.DynSystem.states,
            self.thetaSingle, self.DynSystem.xGoal], [self.ddhdxdx])
        self.ddhdxdeSingleFun = casadi.Function('ddhdxdeSingleFun', [self.DynSystem.states,
            self.thetaSingle, self.DynSystem.xGoal], [self.ddhdxdeSingle])

    def solve(self, initialState, desiredState, thetaInit, paraDict: dict):
        lossTraj = list()
        thetaTraj = list()

        thetaNow = thetaInit
        for idx in range(int(paraDict["maxIter"])):
            
            lossNow, thetaNext = self.gradientDescentVanilla(initialState, desiredState, thetaNow, paraDict)

            lossTraj.append(lossNow)
            thetaTraj.append(thetaNow)

            thetaNow = thetaNext

            if idx % 50 == 0:
                print('Iter:', idx, ' loss:', lossNow)

        resultDict = self.OcSystem.solve(initialState, desiredState, thetaNow)
        lossNow = self.lossFun(resultDict["xi"], thetaNow, desiredState).full()[0, 0]
        lossTraj.append(lossNow)
        thetaTraj.append(thetaNow)

        print('Iter:', paraDict["maxIter"], ' loss:', lossNow)
        print("Theta: ")
        print(thetaNow)

        # visualize
        self.DynSystem.visualize(resultDict, initialState, desiredState, blockFlag=False)
        # plot the loss
        self.plotLossTraj(lossTraj, paraDict["maxIter"])


    def computeGradient(self, initialState, desiredState, thetaNow):
        resultDict = self.OcSystem.solve(initialState, desiredState, thetaNow)
        lqrSystem = self.getLqrSystem(resultDict, initialState, desiredState, thetaNow)
        resultLqr = self.solveLqr(lqrSystem)
        lossNow = self.lossFun(resultDict["xi"], thetaNow, desiredState).full()[0, 0]

        dLdXi = self.dLdXiFun(resultDict["xi"], thetaNow, desiredState)
        dXidTheta = np.vstack((np.concatenate(resultLqr["XTrajList"], axis=0),
            np.concatenate(resultLqr["UTrajList"], axis=0)))
        dLdTheta = self.dLdThetaFun(resultDict["xi"], thetaNow, desiredState)

        gradient = np.array(np.dot(dLdXi, dXidTheta) + dLdTheta).flatten()

        return lossNow, gradient

    def gradientDescentVanilla(self, initialState, desiredState, thetaNow, paraDict: dict):
        lossNow, gradient = self.computeGradient(initialState, desiredState, thetaNow)
        thetaNext = thetaNow - paraDict["stepSize"] * gradient
        return lossNow, thetaNext

    def getLqrSystem(self, resultDict: dict, initialState, desiredState, theta):
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





        # xNow = xTraj[0, :]  # skip the initial condition
        # uNow = uTraj[0, :]
        # lambdaNext = costateTraj[0, :]
        # dynF.append(np.zeros((self.DynSystem.dimStates, self.DynSystem.dimStates)))
        # dynG.append(np.array(self.dfduFun(xNow, uNow, theta).full()))
        # dynE.append(np.array(self.dfdeFun(xNow, uNow, theta).full()))

        # Hxx.append(np.zeros((self.DynSystem.dimStates, self.DynSystem.dimStates)))
        # Hxu.append(np.zeros((self.DynSystem.dimStates, self.DynSystem.dimInputs)))
        # Hxe.append(np.zeros((self.DynSystem.dimStates, self.DynSystem.dimParameters)))
        # Hux.append(np.zeros((self.DynSystem.dimInputs, self.DynSystem.dimStates)))
        
        # Huu.append(np.array(self.ddHduduFun(xNow, uNow, lambdaNext, theta[0], desiredState).full()))

        # HueSingle = np.array(self.ddHdudeSingleFun(xNow, uNow, lambdaNext, theta[0], desiredState).full()).flatten()
        # HueNow = np.zeros((self.DynSystem.dimInputs, self.DynSystem.dimParameters))
        # HueNow[:, 0] = HueSingle
        # Hue.append(HueNow)

        # # compute the matrices, skip the initial condition and terminal state
        # for idx in range(1, uTraj.shape[0]):
        #     xNow = xTraj[idx, :]
        #     uNow = uTraj[idx, :]
        #     lambdaNext = costateTraj[idx, :]  # costate

        #     dynF.append(np.array(self.dfdxFun(xNow, uNow, theta).full()))
        #     dynG.append(np.array(self.dfduFun(xNow, uNow, theta).full()))
        #     dynE.append(np.array(self.dfdeFun(xNow, uNow, theta).full()))

        #     Hxx.append(np.array(self.ddHdxdxFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()))
        #     Hxu.append(np.array(self.ddHdxduFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()))
        #     Hux.append(np.array(self.ddHdudxFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()))
        #     Huu.append(np.array(self.ddHduduFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()))

        #     HxeSingle = np.array(self.ddHdxdeSingleFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()).flatten()
        #     HueSingle = np.array(self.ddHdudeSingleFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()).flatten()

        #     HxeNow = np.zeros((self.DynSystem.dimStates, self.DynSystem.dimParameters))
        #     HueNow = np.zeros((self.DynSystem.dimInputs, self.DynSystem.dimParameters))

        #     HxeNow[:, idx] = HxeSingle
        #     HueNow[:, idx] = HueSingle
        #     Hxe.append(HxeNow)
        #     Hue.append(HueNow)





        # compute the matrices, skip the initial condition and terminal state
        for idx in range(uTraj.shape[0]):
            xNow = xTraj[idx, :]
            uNow = uTraj[idx, :]
            lambdaNext = costateTraj[idx, :]  # costate

            dynF.append(np.array(self.dfdxFun(xNow, uNow, theta).full()))
            dynG.append(np.array(self.dfduFun(xNow, uNow, theta).full()))
            dynE.append(np.array(self.dfdeFun(xNow, uNow, theta).full()))

            Hxx.append(np.array(self.ddHdxdxFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()))
            Hxu.append(np.array(self.ddHdxduFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()))
            Hux.append(np.array(self.ddHdudxFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()))
            Huu.append(np.array(self.ddHduduFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()))

            HxeSingle = np.array(self.ddHdxdeSingleFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()).flatten()
            HueSingle = np.array(self.ddHdudeSingleFun(xNow, uNow, lambdaNext, theta[idx], desiredState).full()).flatten()

            HxeNow = np.zeros((self.DynSystem.dimStates, self.DynSystem.dimParameters))
            HueNow = np.zeros((self.DynSystem.dimInputs, self.DynSystem.dimParameters))

            HxeNow[:, idx] = HxeSingle
            HueNow[:, idx] = HueSingle
            Hxe.append(HxeNow)
            Hue.append(HueNow)





        hxx.append(np.array(self.ddhdxdxFun(xTraj[-1, :], theta[idx], desiredState).full()))
        hxeSingle = np.array(self.ddhdxdeSingleFun(xTraj[-1, :], theta[idx], desiredState).full()).flatten()
        hxeNow = np.zeros((self.DynSystem.dimStates, self.DynSystem.dimParameters))
        hxeNow[:, -1] = hxeSingle
        hxe.append(hxeNow)

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
        I = np.eye(self.DynSystem.dimStates)
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

            tempMat = np.matmul(np.transpose(At), np.linalg.inv(I + np.matmul(PNext, Rt)))
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
                   - np.linalg.multi_dot([invHuu, np.transpose(lqrSystem["dynG"][idx]), np.linalg.inv(I + np.dot(PNext, Rt)),
                   (np.matmul(np.matmul(PNext, At), XNow) + np.matmul(PNext, Mt) + WNext)])

            xNext = np.matmul(lqrSystem["dynF"][idx], XNow) + np.matmul(lqrSystem["dynG"][idx], UNow) + lqrSystem["dynE"][idx]
            XTrajList[idx + 1] = xNext
            UTrajList[idx] = UNow

        result = {"XTrajList": XTrajList,
                  "UTrajList": UTrajList}
        return result

    def plotLossTraj(self, lossTraj, maxIter: int):
        _, ax1 = plt.subplots(1, 1)
        ax1.plot(np.arange(maxIter+1), lossTraj, color="blue", linewidth=2)
        ax1.set_title("Loss")
        ax1.legend(["Loss"])
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("loss")

        plt.show()
