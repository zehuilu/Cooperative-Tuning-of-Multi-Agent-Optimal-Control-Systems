# Cooperative Tuning of Multi-Agent Optimal Control Systems
This is Cooperative Tuning of Multi-Agent Optimal Control Systems. We submitted the paper to IEEE L-CSS with the CDC option.

This repo has been tested with:
* Python 3.9.10, macOS 11.4
* python 3.8.5, Ubuntu 20.04.2 LTS


Dependencies
============

Dependency:
 * [CasADi](https://web.casadi.org/)
 * [cyipopt](https://pypi.org/project/cyipopt/)
 * [NumPy](https://numpy.org/)
 * [Matplotlib](https://matplotlib.org/)


Installation
============
```
$ pip3 install numpy matplotlib casadi cyipopt
$ cd
$ git clone https://github.com/zehuilu/Cooperative-Tuning-of-Multi-Agent-Optimal-Control-Systems.git
```


Usage
=====

To use coopeartive tuning for a user-defined dynamic system, you need to define a class containing the dynamics of the system and the objective funtion, loss function, etc. You can mimic [`Unicycle.py`](/src/Unicycle.py).

NOTE: All the class properties and methods must have the same names as those in [`Unicycle.py`](/src/Unicycle.py). Otherwise, other source files won't be able to use these properties and methods.


Example
=======

The simulation example in our paper is shown in [`run_MultiPDP_Unicycle.py`](/example/run_MultiPDP_Unicycle.py).
```
$ cd <MAIN_DIRECTORY>
$ python3 example/run_MultiPDP_Unicycle.py
```

To test an optimal control with a self-defined dynamic system, see [`run_oc_Unicycle.py`](/example/run_oc_Unicycle.py).
```
$ cd <MAIN_DIRECTORY>
$ python3 example/run_oc_Unicycle.py
```
