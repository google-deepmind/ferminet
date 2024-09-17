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
          # Objective type. One of:
          # 'vmc': minimise <H> by standard VMC energy minimization
          # 'wqmc': minimise <H> by Wasserstein QMC
          # 'vmc_overlap': minimize \sum_i <H_i> + \lambda \sum_ij <psi_i psi_j>
          'objective': 'vmc',
          'iterations': 1000000,  # number of iterations
          'optimizer': 'kfac',  # one of adam, kfac, lamb, none
          'laplacian': 'default',  # of of default or folx (for forward lapl)
          # If 0, use standard vmap. If >0, the max batch size for batched_vmap
          'max_vmap_batch_size': 0,
          'lr': {
              'rate': 0.05,  # learning rate
              'decay': 1.0,  # exponent of learning rate decay
              'delay': 10000.0,  # term that sets the scale of the rate decay
          },
          # If greater than zero, scale (at which to clip local energy) in units
          # of the mean deviation from the mean.
          'clip_local_energy': 5.0,
          # If true, center the clipping window around the median rather than
          # the mean. More "correct" for removing outliers, but also potentially
          # slow, especially with multihost training.
          'clip_median': False,
          # If true, center the local energy differences in the gradient at the
          # average clipped energy rather than average energy, guaranteeing that
          # the average energy difference will be zero in each batch.
          'center_at_clip': True,
          # If true, keep the parameters and optimizer state from the previous
          # step and revert them if they become NaN after an update. Mainly
          # useful for excited states
          'reset_if_nan': False,
          # If using Wasserstein QMC, this parameter controls the amount of
          # "default" VMC gradient to mix in. Otherwise, it is ignored.
          'vmc_weight': 0.0,
          # If nonzero, add a term to the Hamiltonian proportional to the spin
          # magnitude. Useful for removing non-singlet states from excited
          # state calculations.
          'spin_energy': 0.0,
          # If 'objective' is 'vmc_overlap', these parameters control the
          # penalty term.
          'overlap': {
              # Weights on each state. Generate automatically if none provided.
              'weights': None,
              # Strength of the penalty term
              'penalty': 1.0,
          },
          # KFAC hyperparameters. See KFAC documentation for details.
          'kfac': {
              'invert_every': 1,
              'cov_update_every': 1,
              'damping': 0.001,
              'cov_ema_decay': 0.95,
              'momentum': 0.0,
              'momentum_type': 'regular',
              # Warning: adaptive damping is not currently available.
              'min_damping': 1.0e-4,
              'norm_constraint': 0.001,
              'mean_center': True,
              'l2_reg': 0.0,
              'register_only_generic': False,
          },
          # ADAM hyperparameters. See optax documentation for details.
          'adam': {
              'b1': 0.9,
              'b2': 0.999,
              'eps': 1.0e-8,
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
          # Number of excited states. If 0, use normal ground state machinery.
          # If 1, compute ground state using excited state machinery. If >1,
          # compute that many excited states.
          'states': 0,
          # Units of *input* coords of atoms. Either 'bohr' or
          # 'angstrom'. Internally work in a.u.; positions in
          # Angstroms are converged to Bohr.
          'units': 'bohr',
          # If true, use pseudopotentials
          'use_pp': False,
          # Config for pseudopotential if cfg.system.pp is True
          'pp': {
              # If a pseudopotential is used, specify which. Ignored if no
              # pseudopotential is used.
              'type': 'ccecp',
              # If a pseudopotential is used, specify the basis set. Ignored if
              # no pseudopotential is used.
              'basis': 'ccecp-cc-pVDZ',
              # If a pseudopotential is used, list the symbols of elements for
              # which it will be used.
              'symbols': None,
          },
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
          'blocks': 1,  # Number of blocks to split the MCMC sampling into
      },
      'network': {
          'network_type': 'ferminet',  # One of 'ferminet' or 'psiformer'.
          # If true, the network outputs complex numbers rather than real.
          'complex': False,
          # Config specific to original FermiNet architecture.
          # Only used if network_type is 'ferminet'.
          'ferminet': {
              # FermiNet architecture: Pfau, Spencer, Matthews, Foulkes, Phys
              # Rev Research 033429 (2020).
              'hidden_dims': ((256, 32), (256, 32), (256, 32), (256, 32)),
              # Whether to use the last layer of the two-electron stream of the
              # FermiNet.
              'use_last_layer': False,
              # Use separate learnable parameters for pairs of spin-parallel and
              # spin-antiparallel electrons.
              'separate_spin_channels': False,
              # SchNet-style convolutions for permutation-equivariant blocks in
              # FermiNet proposed by Gerard, Scherbela, Marquetand, Grohs,
              # arXiv:2205.09438 (NeurIPS 2022).
              # Dimensions of embeddings for SchNet-style convolution layers
              # (e-e) SchNet-style convolution layers (e-e only) proposed by
              # Gerard et al.  Note: unlike Gerard, we do not currently use
              # separate weights for same-spin and opposite spin interactions
              # unless separate_spin_channels is enabled.  Must be either empty
              # (don't use), or a tuple of embedding dimensions of length N
              # (use_last_layer=False) or N+1 (use_last_layer=True), where N is
              # the number of layers specified in hidden_dims.
              'schnet_electron_electron_convolutions': (),
              # Dimension of each layer of the electron-nuclear auxiliary
              # stream.  Must be same length as hidden_dims-1
              # (use_last_layer=False) or hidden_dims (use_last_layer=True).  If
              # falsy, not used.
              'electron_nuclear_aux_dims': (),
              # Dimensions of the embedding of the nuclear features used in
              # SchNet-style convolutions.  If falsy, not used.
              'nuclear_embedding_dim': 0,
              # Dimensions of embeddings for SchNet-style convolution layers
              # (e-n) proposed by Gerard et al.  Requires
              # electron_nuclear_aux_dims and nuclear_embedding_dim to also be
              # set. Length as for schnet_electron_electron_convolutions.
              'schnet_electron_nuclear_convolutions': (),
          },
          # Only used if network_type is 'psiformer'.
          'psiformer': {
              # PsiFormer architecture: von Glehn, Spencer, Pfau, ICLR 2023.
              'num_layers': 4,
              'num_heads': 4,
              'heads_dim': 64,
              'mlp_hidden_dims': (256,),
              'use_layer_norm': True,
          },
          # Config common to all architectures.
          'determinants': 16,  # Number of determinants.
          'bias_orbitals': False,  # include bias in last layer to orbitals
          # If true, determinants are dense rather than block-sparse
          'full_det': True,
          # If specified, include a pre-determinant Jastrow factor.
          # One of 'default' (use network_type default), 'none', or 'simple_ee'.
          'jastrow': 'default',
          # If true, rescale the inputs so they grow as log(|r|)
          'rescale_inputs': False,
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
          'make_envelope_kwargs': {},
      },
      'observables': {
          's2': False,  # spin magnitude
          'density': False,  # density matrix
          'density_basis': 'def2-tzvpd',  # basis used for DM calculation
          'dipole': False,  # dipole moment
      },
      'debug': {
          # Check optimizer state, parameters and loss and raise an exception if
          # NaN is found.
          'check_nan': False,
          'deterministic': False,  # Use a deterministic seed.
      },
      'pretrain': {
          'method': 'hf',  # Currently only 'hf' is supported.
          'iterations': 1000,  # Only used if method is 'hf'.
          'basis': 'ccpvdz',  # Larger than STO-6G, but good for excited states
          # Fraction of SCF to use in pretraining MCMC. This enables pretraining
          # similar to the original FermiNet paper.
          'scf_fraction': 1.0,
          # The way to construct different states for excited state pretraining.
          # One of 'ordered' or 'random'. 'Ordered' tends to work better, but
          # 'random' is necessary for some systems, especially double
          # excitations.
          'excitation_type': 'ordered',
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
