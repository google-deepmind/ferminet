# FermiNet: Fermionic Neural Networks

An implementation of the algorithm and experiments defined in "Ab-Initio
Solution of the Many-Electron Schroedinger Equation with Deep Neural Networks",
David Pfau, James S. Spencer, Alex G de G Matthews and W.M.C. Foulkes, Phys.
Rev. Research 2, 033429 (2020). FermiNet is a neural network for learning the
ground state wavefunctions of atoms and molecules using a variational Monte
Carlo approach.

## Installation

`pip install -e .` will install all required dependencies. This is best done
inside a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/).

```
virtualenv -p python3.7 ~/venv/ferminet
source ~/venv/ferminet/bin/activate
pip install -e .
```

If you have a GPU available (highly recommended for fast training), then use
`pip install -e '.[tensorflow-gpu]'` to install TensorFlow with GPU support.

We use python 3.7 (or earlier) because there is TensorFlow 1.15 wheel available
for it. TensorFlow 2 is not currently supported.

The tests are easiest run using pytest:

```
pip install pytest
python -m pytest
```

## Usage

```
ferminet --batch_size 1024 --pretrain_iterations 100
```

will train FermiNet to find the ground-state wavefunction of the LiH molecule
with a bond-length of 1.63999 angstroms using a batch size of 1024 MCMC
configurations ("walkers" in variational Monte Carlo language), and 100
iterations of pretraining (the default of 1000 is overkill for such a small
system). The system and hyperparameters can be controlled by flags. Run

```
ferminet --help
```

to see the available options. Several systems used in the FermiNet paper are
included by default. Other systems can easily be set up, by setting the
appropriate system flags to `ferminet`, modifying `ferminet.utils.system` or
writing a custom training script. For example, to run on the H2 molecule:

```
import sys

from absl import logging
from ferminet.utils import system
from ferminet import train

# Optional, for also printing training progress to STDOUT
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# Define H2 molecule
molecule = [system.Atom('H', (0, 0, -1)), system.Atom('H', (0, 0, 1))]

train.train(
  molecule=molecule,
  spins=(1, 1),
  batch_size=256,
  pretrain_config=train.PretrainConfig(iterations=100),
  logging_config=train.LoggingConfig(result_path='H2'),
)
```

`train.train` is controlled by a several lightweight config objects. Only
non-default settings need to be explicitly supplied. Please see the docstrings
for `train.train` and associated `*Config` classes for details.

Note: to train on larger atoms and molecules with large batch sizes, multi-GPU
parallelisation is essential. This is supported via TensorFlow's
[MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
and the `--multi_gpu` flag.

## Output

The results directory contains `pretrain_stats.csv`, which contains the
pretraining loss for each iteration, `train_stats.csv` which contains the local
energy and MCMC acceptance probability for each iteration, and the `checkpoints`
directory, which contains the checkpoints generated during training. If
requested, there is also an HDF5 file, `data.h5`, which contains the walker
configuration, per-walker local energies and per-walker wavefunction values for
each iteration. Warning: this quickly becomes very large!

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
