# Sweeps in EKS and wandb


## How does it work

The refactor removes completely the need to inject hyperparamters on the yaml files. It relies on the sweep controller to do this.

You will need to launch `run-grid.py` from a machine with access to `kubectl`.
