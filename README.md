# PDE-backward-solvers
This is the source code for the experiments from 
http://proceedings.mlr.press/v139/richter21a.html
https://arxiv.org/abs/2102.11830

## Usage
For the NN-experiments, run the Notebooks located in 'experiments'

For the TT-solver, script.py, located in the TT_experiments folder, should be executed. You can choose between
different premade setups (from the paper) or make a custom setup by setting the 'setup' parameter (l. 29 in
script.py)

Warning: The BSDE solution and also some other matrices are saved during the
optimization. Thus, some disc space has to be free. From time to time it is
advised to run 'rm V_*', 'rm c_*', 'rm Y_*' in the TT_experiments folder.









## Dependencies

- xerus (SALSA branch) https://libxerus.org/
- pytorch
- numpy
- scipy
- matplotlib
