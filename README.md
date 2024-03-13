# Online-3D-IBPP

Part of the Thesis *"Learning Online 3d Irregular Bin Packing On Packing Configuration Trees"*, by Thomas Vroom for the Bachelor Data Science and Artificial Intelligence at Maastricht University.

## Introduction

Online 3D Irregular Bin Packing Problems (3DIBPP) are some of the most challenging, but practical packing problems that come up in industrial settings.
In these problems, the goal is to move a sequence of irregularly-shaped objects into a 3D container, without knowing what object comes next in the sequence [(Xu H. Z., 2022)](https://arxiv.org/abs/2006.14978).
Although this is classically a combinatorial optimization problem, research has shown that reinforcement learning can be used to effectively approximate bin packing problems, and even out-perform non-learning based methods [(Xu H. H., 2017)](https://arxiv.org/abs/1708.05930).

Recently, [Zhao et al. (2022)](https://openreview.net/forum?id=bfuGjlCwAq) introduced a novel way to represent the state and action space of packing problems, called Packing Configuration Trees (PCT).
A PCT is a hierarchical representation of the state and action space of packing problems, where the size of the action space is proportional to the number of leaf nodes.
The traversal through a PCT is formulated as a policy that can be learned through deep reinforcement learning (DRL).
During training, the PCT expands based on heuristic rules, however, research has shown that combining a PCT with a DRL model results in a more effective packing policy than heuristic methods [(Xu H. Z., 2022)](https://arxiv.org/abs/2212.02094).
Furthermore, [Zhao et al. (2022)](https://openreview.net/forum?id=bfuGjlCwAq) also showed that PCT with DRL outperformed all existing online 3D packing methods and is versatile in terms of incorporating various practical constraints.

The goal of this thesis is to take the framework of PCT combined with DRL, and to extend it to online 3DIBPP.

## Dependencies
See `requirements.txt`.

## Usage
...

## Acknowledgements
This repository uses modified code from [Zhao et al. (2022)](https://github.com/alexfrom0815/Online-3D-BPP-PCT).

## License
[MIT](https://choosealicense.com/licenses/mit/)
