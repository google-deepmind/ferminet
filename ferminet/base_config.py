# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default base configuration for molecular VMC calculations."""

import enum

import ml_collections
from ml_collections import config_dict


class SystemType(enum.IntEnum):
  """Enum for system types.

  WARNING: enum members cannot be serialised readily so use
  SystemType.member.value in such cases.
  """
  MOLECULE = 0

  @classmethod
  def has_value(cls, value):
    return any(value is item or value == item.value for item in cls)


def default() -> ml_collections.ConfigDict:
  """Create set of default parameters for running qmc.py.

  Note: placeholders (cfg.system.molecule and cfg.system.electrons) must be
  replaced with appropriate values.

  Returns:
    ml_collections.ConfigDict containing default settings.
  """
  # wavefunction output.
  cfg = ml_collections.ConfigDict({
      'batch_size': 4096,  # batch size
      # Config module used. Should be set in get_config function as either the
      # absolute module or relative to the configs subdirectory. Relative
      # imports must start with a '.' (e.g. .atom). Do *not* override on
      # command-line. Do *not* set using __name__ from inside a get_config
      # function, as config_flags overrides this when importing the module using
      # importlib.import_module.
      'config_module': __name__,
      'optim': {
          'iterations': 1000000,  # number of iterations
          'optimizer': 'kfac',  # one of adam, kfac, lamb, none
          'lr': {
              'rate': 1.e-4,  # learning rate
              'decay': 1.0,  # exponent of learning rate decay
              'delay': 10000.0,  # term that sets the scale of the rate decay
          },
          'clip_el': 5.0,  # If not none, scale at which to clip local energy
          # KFAC hyperparameters. See KFAC documentation for details.
          'kfac': {
              'invert_every': 1,
              'cov_update_every': 1,
              'damping': 0.001,
              'cov_ema_decay': 0.95,
              'momentum': 0.0,
              'momentum_type': 'regular',
              # Warning: adaptive damping is not currently available.
              'min_damping': 1.e-4,
              'norm_constraint': 0.001,
              'mean_center': True,
              'l2_reg': 0.0,
              'register_only_generic': False,
          },
          # ADAM hyperparameters. See optax documentation for details.
          'adam': {
              'b1': 0.9,
              'b2': 0.999,
              'eps': 1.e-8,
              'eps_root': 0.0,
          },
      },
      'log': {
          'stats_frequency': 1,  # iterations between logging of stats
          'save_frequency': 10.0,  # minutes between saving network params
          # Path to save/restore network to/from. If falsy,
          # creates a timestamped directory in the working directory.
          'save_path': '',
          # Path containing checkpoint to restore network from.
          # Ignored if falsy or save_path contains a checkpoint.
          'restore_path': '',
          # Remaining log options are currently not functional.  Whether or not
          # to log the values of all walkers every iteration Use with caution!!!
          # Produces a lot of data very quickly.
          'walkers': False,
          # Whether or not to log all local energies for each walker at each
          # step
          'local_energies': False,
          # Whether or not to log all values of wavefunction or log abs
          # wavefunction dependent on using log_energy mode or not for each
          # walker at each step
          'features': False,
      },
      'system': {
          'type': SystemType.MOLECULE.value,
          # Alternatively, set 'set_molecule' to a function which takes the
          # ConfigDict as the sole argument and sets the molecule value (and any
          # other related values) in the ConfigDict.
          'molecule': config_dict.placeholder(list),
          # Users can pass in a pyscf_mol here, instead of the internal
          # representation.
          'pyscf_mol': None,
          'electrons': tuple(),  # electrons in system
          # Change with care. FermiNet implementation currently assumes 3D
          # systems.
          'ndim': 3,
          'units': 'bohr',  # Units of *input* coords of atoms. Either 'bohr' or
          # 'angstrom'. Internally work in a.u.; positions in
          # Angstroms are converged to Bohr.
          'set_molecule': None,  # Callable[ConfigDict] -> ConfigDict.
          # Takes a ConfigDict and sets molecule and
          # other related values. Returns the ConfigDict
          # with these set. Note: modifications may also
          # be performed in-place.
      },
      'mcmc': {
          # Note: HMC options are not currently used.
          # Number of burn in steps after pretraining.  If zero do not burn in
          # or reinitialize walkers.
          'burn_in': 100,
          'steps': 10,  # Number of MCMC steps to make between network updates.
          # Width of (atom-centred) Gaussian used to generate initial electron
          # configurations.
          'init_width': 0.8,
          # Width of Gaussian used for random moves for RMW
          'move_width': 0.02,
          # Number of steps after which to update the adaptive MCMC step size
          'adapt_frequency': 100,
          'use_hmc': False,  # Use HMC (True) or Random Walk Metropolis (False)
          # Size of hmc leapfrog steps. Unused if not using HMC
          'step_size': .0005,
          'path_len': .002,  # Leapfrog path len. Unused if not using HMC
          'hmc_steps': 5,  # Number of HMC outer steps. Unused if not using HMC
          # Iterable of 3*nelectrons giving the mean initial position of each
          # electron. Configurations are drawn using Gaussians of width
          # init_width at each 3D position. Alpha electrons are listed before
          # beta electrons. If falsy, electrons are assigned to atoms based upon
          # the isolated atom spin configuration.
          'init_means': (),  # Not implemented in JAX.
          # If true, scale the proposal width for each electron by the harmonic
          # mean of the distance to the nuclei.
          'scale_by_nuclear_distance': False,
          'one_electron': False  # If true, use one-electron moves
      },
      'network': {
          'detnet': {
              'hidden_dims': ((256, 32), (256, 32), (256, 32), (256, 32)),
              'determinants': 16,
              'after_determinants': (1,),
          },
          'envelope_type': 'full',  # Where does the envelope go?
          'bias_orbitals': False,  # include bias in last layer to orbitals
          # Whether to use the last layer of the two-electron stream of the
          # DetNet
          'use_last_layer': False,
          # If true, determinants are dense rather than block-sparse
          'full_det': True,
      },
      'debug': {
          # Check optimizer state, parameters and loss and raise an exception if
          # NaN is found.
          'check_nan': False,
          'deterministic': False,  # Use a deterministic seed.
      },
      'pretrain': {
          'method': 'hf',  # Method is one of 'hf', or 'direct_init'.
          'iterations': 1000,  # Only used if method is 'hf'.
          'basis': 'sto-6g',
      },
  })

  return cfg


def resolve(cfg):
  """Resolve any ml_collections.config_dict.FieldReference values in a ConfigDict for qmc.

  Any FieldReferences in the coords array for each element in
  cfg.system.molecule are treated specially as nested references are not
  resolved by ConfigDict.copy_and_resolve_references. Similar cases should be
  added here as needed.

  Args:
    cfg: ml_collections.ConfigDict containing settings.

  Returns:
    ml_collections.ConfigDict with ml_collections.FieldReference values resolved
    (as far as possible).

  Raises:
    RuntimeError: If an atomic position is non-numeric.
  """
  if 'set_molecule' in cfg.system and callable(cfg.system.set_molecule):
    cfg = cfg.system.set_molecule(cfg)
    with cfg.ignore_type():
      # Replace the function with its name so we know how the molecule was set
      # This makes the ConfigDict object serialisable.
      if callable(cfg.system.set_molecule):
        cfg.system.set_molecule = cfg.system.set_molecule.__name__
  cfg = cfg.copy_and_resolve_references()
  return cfg
