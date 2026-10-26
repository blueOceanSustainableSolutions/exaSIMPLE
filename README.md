# exaSIMPLE

- Use of graph neural networks provides a way to solve the matrix equation systems using GPUs
- Approximation of SIMPLE algorithm terms that disregarded in the traditional approaches lead to a considerable improvement of performance, accuracy and robustness for several canonical, benchmark and simple Industrial-typical CFD cases
- exaSIMPLE code permits all CFD practitioners to reproduce the results here obtained and transfer the major findings to any CFD code

`L1_exaMLSolver` contains the full codebase for the exaSOLVER GNN pipeline

`L2` contains the full codebase for the pressure-velocity coupling improvements

Detailed user guides for each of these are contained in their respective repositories. 

## Acknowledgements                                                                                                                                                 
This work was carried out on the Deucalion supercomputer (MACC, Guimarães, Portugal).
We acknowledge the Portuguese Foundation for Science and Technology (FCT) for providing 
the computing resources through the Advanced Computing Project 2024.14686.CPCA.A3.                                                                                                   

GNN and neural sparse linear solver models were trained on Deucalion's NVIDIA A100 (80 GB) 
nodes: more than 360 jobs and about 2,200 A100 node-hours between November 2024 and 
February 2025, including multi-GPU (FSDP) training and Optuna hyperparameter searches.                                                                                             
We thank the Deucalion support team at CNCA for their help.
