# Cooperative Tuning of Multi-Agent Optimal Control Systems
This is Cooperative Tuning of Multi-Agent Optimal Control Systems. This paper was accepted by the 61st IEEE Conference on Decision and Control, 2022. The arxiv version is
https://arxiv.org/abs/2209.12017

Please cite us if you think this work is helpful. An IEEE version of citation information will be updated once it's available online.
```
@misc{https://doi.org/10.48550/arxiv.2209.12017,
  doi = {10.48550/ARXIV.2209.12017},
  url = {https://arxiv.org/abs/2209.12017},
  author = {Lu, Zehui and Jin, Wanxin and Mou, Shaoshuai and Anderson, Brian D. O.},
  title = {Cooperative Tuning of Multi-Agent Optimal Control Systems},
  publisher = {arXiv},
  year = {2022},
}
```

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

To test an optimal control with a self-defined dynamic system, see [`run_oc_Unicycle.py`](/example/run_oc_Unicycle.py).
```
$ cd <MAIN_DIRECTORY>
$ python3 example/run_oc_Unicycle.py
```

The simulation example in our paper is shown in [`run_MultiPDP_Unicycle.py`](/example/run_MultiPDP_Unicycle.py).
```
$ cd <MAIN_DIRECTORY>
$ python3 example/run_MultiPDP_Unicycle.py
```
