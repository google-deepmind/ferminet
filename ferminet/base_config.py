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
  MOLECULE = enum.auto()

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
              'rate': 0.05,  # learning rate
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
          # Specify the system.
          # 1. Specify the system by setting variables below.
          # list of system.Atom objects with element type and position.
          'molecule': config_dict.placeholder(list),
          # number of spin up, spin-down electrons
          'electrons': tuple(),
          # Dimensionality. Change with care. FermiNet implementation currently
          # assumes 3D systems.
          'ndim': 3,
          # Units of *input* coords of atoms. Either 'bohr' or
          # 'angstrom'. Internally work in a.u.; positions in
          # Angstroms are converged to Bohr.
          'units': 'bohr',
          # 2. Specify the system using pyscf. Must be a pyscf.gto.Mole object.
          'pyscf_mol': None,
          # 3. Specify the system inside a function evaluated after the config
          # has been parsed.
          # Callable[ConfigDict] -> ConfigDict which sets molecule and
          # other related values and returns the ConfigDict with these set.
          # Note: modifications may also be performed in-place.
          'set_molecule': None,
          # String set to module.make_local_energy, where make_local_energy is a
          # callable (type: MakeLocalEnergy) which creates a function which
          # evaluates the local energy and module is the absolute module
          # containing make_local_energy.
          # If not set, hamiltonian.local_energy is used.
          'make_local_energy_fn': '',
          # Additional kwargs to pass into make_local_energy_fn.
          'make_local_energy_kwargs': {},
      },
      'mcmc': {
          # Note: HMC options are not currently used.
          # Number of burn in steps after pretraining.  If zero do not burn in
          # or reinitialize walkers.
          'burn_in': 100,
          'steps': 10,  # Number of MCMC steps to make between network updates.
          # Width of (atom-centred) Gaussian used to generate initial electron
          # configurations.
          'init_width': 1.0,
          # Width of Gaussian used for random moves for RMW or step size for
          # HMC.
          'move_width': 0.02,
          # Number of steps after which to update the adaptive MCMC step size
          'adapt_frequency': 100,
          'use_hmc': False,  # Use HMC (True) or Random Walk Metropolis (False)
          # Number of HMC leapfrog steps.  Unused if not doing HMC.
          'num_leapfrog_steps': 10,
          # Iterable of 3*nelectrons giving the mean initial position of each
          # electron. Configurations are drawn using Gaussians of width
          # init_width at each 3D position. Alpha electrons are listed before
          # beta electrons. If falsy, electrons are assigned to atoms based upon
          # the isolated atom spin configuration.
          'init_means': (),  # Not implemented in JAX.
          # If true, scale the proposal width for each electron by the harmonic
          # mean of the distance to the nuclei.
          'scale_by_nuclear_distance': False,
          'one_electron': False,  # If true, use one-electron moves
      },
      'network': {
          'detnet': {
              'hidden_dims': ((256, 32), (256, 32), (256, 32), (256, 32)),
              'determinants': 16,
              'after_determinants': (1,),
          },
          'bias_orbitals': False,  # include bias in last layer to orbitals
          # Whether to use the last layer of the two-electron stream of the
          # DetNet
          'use_last_layer': False,
          # If true, determinants are dense rather than block-sparse
          'full_det': True,
          # String set to module.make_feature_layer, where make_feature_layer is
          # callable (type: MakeFeatureLayer) which creates an object with
          # member functions init() and apply() that initialize parameters
          # for custom input features and modify raw input features,
          # respectively. Module is the absolute module containing
          # make_feature_layer.
          # If not set, networks.make_ferminet_features is used.
          'make_feature_layer_fn': '',
          # Additional kwargs to pass into make_local_energy_fn.
          'make_feature_layer_kwargs': {},
          # Same structure as make_feature_layer
          'make_envelope_fn': '',
          'make_envelope_kwargs': {}
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
