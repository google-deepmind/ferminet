# FermiNet: Fermionic Neural Networks

FermiNet is a neural network for learning highly accurate ground state
wavefunctions of atoms and molecules using a variational Monte Carlo approach.

This repository contains an implementation of the algorithm and experiments
first described in "Ab-Initio Solution of the Many-Electron Schroedinger
Equation with Deep Neural Networks", David Pfau, James S. Spencer, Alex G de G
Matthews and W.M.C. Foulkes, Phys. Rev. Research 2, 033429 (2020), along with
subsequent research and developments.

WARNING: This is a research-level release of a JAX implementation and is under
active development. The original TensorFlow implementation can be found in the
`tf` branch.

## Installation

`pip install -e .` will install all required dependencies. This is best done
inside a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/).

```shell
virtualenv ~/venv/ferminet
source ~/venv/ferminet/bin/activate
pip install -e .
```

If you have a GPU available (highly recommended for fast training), then you can
install JAX with CUDA support, using e.g.:

```shell
pip install --upgrade jax jaxlib==0.1.57+cuda110 -f
https://storage.googleapis.com/jax-releases/jax_releases.html
```

Note that the jaxlib version must correspond to the existing CUDA installation
you wish to use. Please see the
[JAX documentation](https://github.com/google/jax#installation) for more
details.

The tests are easiest run using pytest:

```shell
pip install -e '.[testing]'
python -m pytest
```

## Usage

ferminet uses the `ConfigDict` from
[ml_collections](https://github.com/google/ml_collections) to configure the
system. A few example scripts are included under `ferminet/configs/`. These are
mostly for testing so may need additional settings for a production-level
calculation.

```shell
ferminet --config ferminet/configs/atom.py --config.system.atom Li --config.batch_size 256 --config.pretrain.iterations 100
```

will train FermiNet to find the ground-state wavefunction of the Li atom using a
batch size of 1024 MCMC configurations ("walkers" in variational Monte Carlo
language), and 100 iterations of pretraining (the default of 1000 is overkill
for such a small system). The system and hyperparameters can be controlled by
modifying the config file or (better, for one-off changes) using flags. See the
[ml_collections](https://github.com/google/ml_collections)' documentation for
further details on the flag syntax. Details of all available config settings are
in `ferminet/base_config.py`.

Other systems can easily be set up, by creating a new config file or `ferminet`,
or writing a custom training script. For example, to run on the H2 molecule, you
can create a config file containing:

```python
from ferminet import base_config
from ferminet.utils import system

# Settings in a config files are loaded by executing the the get_config
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

```python
import sys

from absl import logging
from ferminet.utils import system
from ferminet import base_config
from ferminet import train

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# Define H2 molecule
cfg = base_config.default()
cfg.system.electrons = (1,1)  # (alpha electrons, beta electrons)
cfg.system.molecule = [system.Atom('H', (0, 0, -1)), system.Atom('H', (0, 0, 1))]

# Set training parameters
cfg.batch_size = 256
cfg.pretrain.iterations = 100

train.train(cfg)
```

Alternatively, you can directly pass in a PySCF ['Molecule'](http://pyscf.org).
You can create PySCF Molecules with the following:

```python
from pyscf import gto
mol = gto.Mole()
mol.build(
    atom = 'H  0 0 1; H 0 0 -1',
    basis = 'sto-3g', unit='bohr')
```

Once you have this molecule, you can pass it directly into the configuration by
running

```python
from ferminet import base_config
from ferminet import train

# Add H2 molecule
cfg = base_config.default()
cfg.system.pyscf_mol = mol

# Set training parameters
cfg.batch_size = 256
cfg.pretrain.iterations = 100

train.train(cfg)
```

Note: to train on larger atoms and molecules with large batch sizes, multi-GPU
parallelisation is essential. This is supported via JAX's
[pmap](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap).
Multiple GPUs will be automatically detected and used if available.

## Output

The results directory contains `train_stats.csv` which contains the local energy
and MCMC acceptance probability for each iteration, and the `checkpoints`
directory, which contains the checkpoints generated during training.

## Pretrained Models

A collection of pretrained models trained with KFAC can be found on Google Cloud
[here](https://console.cloud.google.com/storage/browser/dm-ferminet/models).
These are all systems from the original PRResearch paper: carbon and neon atoms,
and nitrogen, ethene, methylamine, ethanol and bicyclobutane molecules. Each
folder contains samples from the wavefunction in `walkers.npy`, parameters in
`parameters.npz` and geometries for the molecule in `geometry.npz`. To load the
models and evaluate the local energy, run:

```python
import numpy as np
import jax
from functools import partial
from ferminet import networks, train

with open('params.npz', 'rb') as f:
  params = dict(np.load(f, allow_pickle=True))
  params = params['arr_0'].tolist()

with open('walkers.npy', 'rb') as f:
  data = np.load(f)

with open('geometry.npz', 'rb') as f:
  geometry = dict(np.load(f, allow_pickle=True))

signed_network = partial(networks.fermi_net, envelope_type='isotropic', full_det=False, **geometry)
# networks.fermi_net gives the sign/log of the wavefunction. We only care about the latter.
network = lambda p, x: signed_network(p, x)[1]
batch_network = jax.vmap(network, (None, 0), 0)
loss = train.make_loss(network, batch_network, geometry['atoms'], geometry['charges'], clip_local_energy=5.0)

print(loss(params, data)[0])  # For neon, should give -128.94165
```

## Giving Credit

If you use this code in your work, please cite the associated papers. The
initial paper details the architecture and results on a range of systems:

```
@article{pfau2020ferminet,
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

and a NeurIPS Workshop Machine Learning and Physics paper describes the JAX
implementation:

```
@misc{spencer2020better,
  title={Better, Faster Fermionic Neural Networks},
  author={James S. Spencer and David Pfau and Aleksandar Botev and W. M.C. Foulkes},
  year={2020},
  eprint={2011.07125},
  archivePrefix={arXiv},
  primaryClass={physics.comp-ph},
  url={https://arxiv.org/abs/2011.07125}
}
```

The PsiFormer architecture is detailed in an ICLR 2023 paper, preprint
reference:

```
@misc{vonglehn2022psiformer,
  title={A Self-Attention Ansatz for Ab-initio Quantum Chemistry},
  author={Ingrid von Glehn and James S Spencer and David Pfau},
  year={2022},
  eprint={2211.13672},
  archivePrefix={arXiv},
  primaryClass={physics.chem-ph},
  url={https://arxiv.org/abs/2211.13672},
}
```

This repository can be cited using:

```
@software{ferminet_github,
  author = {James S. Spencer, David Pfau and FermiNet Contributors},
  title = {{FermiNet}},
  url = {http://github.com/deepmind/ferminet},
  year = {2020},
}
```

## Disclaimer

This is not an official Google product.
