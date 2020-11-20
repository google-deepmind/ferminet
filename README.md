# FermiNet: Fermionic Neural Networks

An implementation of the algorithm and experiments defined in "Ab-Initio
Solution of the Many-Electron Schroedinger Equation with Deep Neural Networks",
David Pfau, James S. Spencer, Alex G de G Matthews and W.M.C. Foulkes, Phys.
Rev. Research 2, 033429 (2020). FermiNet is a neural network for learning the
ground state wavefunctions of atoms and molecules using a variational Monte
Carlo approach.

WARNING: This is a research-level release of a JAX implementation.

## Installation

`pip install -e .` will install all required dependencies. This is best done
inside a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/).

```
virtualenv ~/venv/ferminet
source ~/venv/ferminet/bin/activate
pip install -e .
```

If you have a GPU available (highly recommended for fast training), then you can
install JAX with CUDA support, using e.g.:

```
pip install --upgrade jax jaxlib==0.1.57+cuda110 -f
https://storage.googleapis.com/jax-releases/jax_releases.html
```

Note that the jaxlib version must correspond to the existing CUDA installation
you wish to use. Please see the
[JAX documentation](https://github.com/google/jax#installation) for more
details.

The tests are easiest run using pytest:

```
pip install pytest
python -m pytest
```

## Usage

ferminet uses the `ConfigDict` from
[ml_collections](https://github.com/google/ml_collections) to configure the
system. A few example scripts are included under `ferminet/configs/`. These are
mostly for testing so need additional

```
ferminet --config ferminet/configs/atom.py --config.system.atom Li --config.batch_size 256 --config.pretrain.iterations 100
```

will train FermiNet to find the ground-state wavefunction of the Li atom with a
bond-length of 1.63999 angstroms using a batch size of 1024 MCMC configurations
("walkers" in variational Monte Carlo language), and 100 iterations of
pretraining (the default of 1000 is overkill for such a small system). The
system and hyperparameters can be controlled by modifying the config file or
(better, for one-off changes) using flags. See the
[ml_collections](https://github.com/google/ml_collections)' documentation for
further details on the flag syntax. Details of all available config settings are
in `ferminet/base_config.py`.

Other systems can easily be set up, by creating a new config file or `ferminet`,
or writing a custom training script. For example, to run on the H2 molecule, you
can create a config file containing:

```
from ferminet import base_config
from ferminet.utils import system

# Settings in a a config files are loaded by executing the the get_config
# function.
def get_config():
  # Get default options.
  cfg = base_config.default()
  # Set up molecule
  cfg.system.electrons = (1,1)
  cfg.system.molecule = [system.Atom('H', (0, 0, -1)), system.Atom('H', (0, 0, 1))]

  # Set training hyperparameters
  cfg.batch_size = 256
  cfg.pretrain.iterations = 100

  return cfg
```

and then run it using

```
ferminet --config /path/to/h2_config.py
```

or equivalently write the following script (or execute it interactively):

```
import sys

from absl import logging
from ferminet.utils import system
from ferminet import base_config
from ferminet import qmc

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# Define H2 molecule
cfg = base_config.default()
cfg.system.electrons = (1,1)
cfg.system.molecule = [system.Atom('H', (0, 0, -1)), system.Atom('H', (0, 0, 1))]

# Set training parameters
cfg.batch_size = 256
cfg.pretrain.iterations = 100

qmc.train(cfg)
```

Note: to train on larger atoms and molecules with large batch sizes, multi-GPU
parallelisation is essential. This is supported via JAX's
[pmap](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap).
Multiple GPUs will be automatically detected and used if available.

## Output

The results directory contains `train_stats.csv` which contains the local energy
and MCMC acceptance probability for each iteration, and the `checkpoints`
directory, which contains the checkpoints generated during training.

## Giving Credit

If you use this code in your work, please cite the associated paper:

```
@article{ferminet,
  title={Ab-Initio Solution of the Many-Electron Schr{\"o}dinger Equation with Deep Neural Networks},
  author={D. Pfau and J.S. Spencer and A.G. de G. Matthews and W.M.C. Foulkes},
  journal={Phys. Rev. Research},
  year={2020},
  volume={2},
  issue = {3},
  pages={033429},
  doi = {10.1103/PhysRevResearch.2.033429},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.2.033429}
}
```

## Disclaimer

This is not an official Google product.
