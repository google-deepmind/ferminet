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

"""Core training loop for neural QMC in JAX."""

import functools
import importlib
import os
import time
from typing import Optional, Mapping, Sequence, Tuple, Union

from absl import logging
import chex
from ferminet import checkpoint
from ferminet import constants
from ferminet import curvature_tags_and_blocks
from ferminet import envelopes
from ferminet import hamiltonian
from ferminet import loss as qmc_loss_functions
from ferminet import mcmc
from ferminet import networks
from ferminet import observables
from ferminet import pretrain
from ferminet import psiformer
from ferminet.utils import statistics
from ferminet.utils import system
from ferminet.utils import utils
from ferminet.utils import writers
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol


def _assign_spin_configuration(
    nalpha: int, nbeta: int, batch_size: int = 1
) -> jnp.ndarray:
  """Returns the spin configuration for a fixed spin polarisation."""
  spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
  return jnp.tile(spins[None], reps=(batch_size, 1))


def init_electrons(  # pylint: disable=dangerous-default-value
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width: float,
    core_electrons: Mapping[str, int] = {},
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.
    core_electrons: mapping of element symbol to number of core electrons
      included in the pseudopotential.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3), and array of (batch_size, (nalpha+nbeta))
    of spin configurations, where 1 and -1 indicate alpha and beta electrons
    respectively.
  """
  total_electrons = sum(atom.charge - core_electrons.get(atom.symbol, 0)
                        for atom in molecule)
  if total_electrons != sum(electrons):
    if len(molecule) == 1:
      atomic_spin_configs = [electrons]
    else:
      raise NotImplementedError('No initialization policy yet '
                                'exists for charged molecules.')
  else:
    atomic_spin_configs = [
        (atom.element.nalpha - core_electrons.get(atom.symbol, 0) // 2,
         atom.element.nbeta - core_electrons.get(atom.symbol, 0) // 2)
        for atom in molecule
    ]
    assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
    while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
      i = np.random.randint(len(atomic_spin_configs))
      nalpha, nbeta = atomic_spin_configs[i]
      atomic_spin_configs[i] = nbeta, nalpha

  # Assign each electron to an atom initially.
  electron_positions = []
  for i in range(2):
    for j in range(len(molecule)):
      atom_position = jnp.asarray(molecule[j].coords)
      electron_positions.append(
          jnp.tile(atom_position, atomic_spin_configs[j][i]))
  electron_positions = jnp.concatenate(electron_positions)
  # Create a batch of configurations with a Gaussian distribution about each
  # atom.
  key, subkey = jax.random.split(key)
  electron_positions += (
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
      * init_width
  )

  electron_spins = _assign_spin_configuration(
      electrons[0], electrons[1], batch_size
  )

  return electron_positions, electron_spins


# All optimizer states (KFAC and optax-based).
OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: optax.OptState,
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: electron positions, spins and atomic positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[
    networks.FermiNetData,
    networks.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
    jnp.ndarray,
]


class Step(Protocol):

  def __call__(
      self,
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations, spins and atomic positions.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
    """


def null_update(
    params: networks.ParamTree,
    data: networks.FermiNetData,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
) -> OptUpdateResults:
  """Performs an identity operation with an OptUpdate interface."""
  del data, key
  return params, opt_state, jnp.zeros(1), None


def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
  """Returns an OptUpdate function for performing a parameter update."""

  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters using optax."""
    (loss, aux_data), grad = loss_and_grad(params, key, data)
    grad = constants.pmean(grad)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux_data

  return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
  """Returns an OptUpdate function for evaluating the loss."""

  def loss_eval(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates just the loss and gradients with an OptUpdate interface."""
    loss, aux_data = evaluate_loss(params, key, data)
    return params, opt_state, loss, aux_data

  return loss_eval


def make_training_step(
    mcmc_step,
    optimizer_step: OptUpdate,
    reset_if_nan: bool = False,
) -> Step:
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: Optional[optax.OptState],
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

    # Optimization step
    new_params, new_state, loss, aux_data = optimizer_step(params,
                                                           data,
                                                           state,
                                                           loss_key)
    if reset_if_nan:
      new_params = jax.lax.cond(jnp.isnan(loss),
                                lambda: params,
                                lambda: new_params)
      new_state = jax.lax.cond(jnp.isnan(loss),
                               lambda: state,
                               lambda: new_state)
    return data, new_params, new_state, loss, aux_data, pmove

  return step


def make_kfac_training_step(
    mcmc_step,
    damping: float,
    optimizer: kfac_jax.Optimizer,
    reset_if_nan: bool = False) -> Step:
  """Factory to create traning step for KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    damping: value of damping to use for each KFAC update step.
    optimizer: KFAC optimizer instance.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
  shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
  shared_damping = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(damping))
  # Due to some KFAC cleverness related to donated buffers, need to do this
  # to make state resettable
  copy_tree = constants.pmap(
      functools.partial(jax.tree_util.tree_map,
                        lambda x: (1.0 * x).astype(x.dtype)))

  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: kfac_jax.Optimizer.State,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.

    # MCMC loop
    mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
    data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

    if reset_if_nan:
      old_params = copy_tree(params)
      old_state = copy_tree(state)

    # Optimization step
    new_params, new_state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        batch=data,
        momentum=shared_mom,
        damping=shared_damping,
    )

    if reset_if_nan and jnp.isnan(stats['loss']):
      new_params = old_params
      new_state = old_state
    return data, new_params, new_state, stats['loss'], stats['aux'], pmove

  return step


def train(cfg: ml_collections.ConfigDict, writer_manager=None):
  """Runs training loop for QMC.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details.
    writer_manager: context manager with a write method for logging output. If
      None, a default writer (ferminet.utils.writers.Writer) is used.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
  # Device logging
  num_devices = jax.local_device_count()
  num_hosts = jax.device_count() // num_devices
  num_states = cfg.system.get('states', 0) or 1  # avoid 0/1 confusion
  logging.info('Starting QMC with %i XLA devices per host '
               'across %i hosts.', num_devices, num_hosts)
  if cfg.batch_size % (num_devices * num_hosts) != 0:
    raise ValueError('Batch size must be divisible by number of devices, '
                     f'got batch size {cfg.batch_size} for '
                     f'{num_devices * num_hosts} devices.')
  host_batch_size = cfg.batch_size // num_hosts  # batch size per host
  total_host_batch_size = host_batch_size * num_states
  device_batch_size = host_batch_size // num_devices  # batch size per device
  data_shape = (num_devices, device_batch_size)

  # Check if mol is a pyscf molecule and convert to internal representation
  if cfg.system.pyscf_mol:
    cfg.update(
        system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))

  # Convert mol config into array of atomic positions and charges
  atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  nspins = cfg.system.electrons

  # Generate atomic configurations for each walker
  batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
  batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
  batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
  batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)

  if cfg.debug.deterministic:
    seed = 23
  else:
    seed = jnp.asarray([1e6 * time.time()])
    seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
  key = jax.random.PRNGKey(seed)

  # extract number of electrons of each spin around each atom removed because
  # of pseudopotentials
  if cfg.system.pyscf_mol:
    cfg.system.pyscf_mol.build()
    core_electrons = {
        atom: ecp_table[0]
        for atom, ecp_table in cfg.system.pyscf_mol._ecp.items()  # pylint: disable=protected-access
    }
    ecp = cfg.system.pyscf_mol.ecp
  else:
    ecp = {}
    core_electrons = {}

  # Create parameters, network, and vmaped/pmaped derivations

  if cfg.pretrain.method == 'hf' and cfg.pretrain.iterations > 0:
    hartree_fock = pretrain.get_hf(
        pyscf_mol=cfg.system.get('pyscf_mol'),
        molecule=cfg.system.molecule,
        nspins=nspins,
        restricted=False,
        basis=cfg.pretrain.basis,
        ecp=ecp,
        core_electrons=core_electrons,
        states=cfg.system.states,
        excitation_type=cfg.pretrain.get('excitation_type', 'ordered'))
    # broadcast the result of PySCF from host 0 to all other hosts
    hartree_fock.mean_field.mo_coeff = multihost_utils.broadcast_one_to_all(
        hartree_fock.mean_field.mo_coeff
    )

  if cfg.network.make_feature_layer_fn:
    feature_layer_module, feature_layer_fn = (
        cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
    feature_layer_module = importlib.import_module(feature_layer_module)
    make_feature_layer: networks.MakeFeatureLayer = getattr(
        feature_layer_module, feature_layer_fn
    )
    feature_layer = make_feature_layer(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
        **cfg.network.make_feature_layer_kwargs)
  else:
    feature_layer = networks.make_ferminet_features(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
    )

  if cfg.network.make_envelope_fn:
    envelope_module, envelope_fn = (
        cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
    envelope_module = importlib.import_module(envelope_module)
    make_envelope = getattr(envelope_module, envelope_fn)
    envelope = make_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
  else:
    envelope = envelopes.make_isotropic_envelope()

  use_complex = cfg.network.get('complex', False)
  if cfg.network.network_type == 'ferminet':
    network = networks.make_fermi_net(
        nspins,
        charges,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        states=cfg.system.states,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get('jastrow', 'default'),
        bias_orbitals=cfg.network.bias_orbitals,
        full_det=cfg.network.full_det,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
        complex_output=use_complex,
        **cfg.network.ferminet,
    )
  elif cfg.network.network_type == 'psiformer':
    network = psiformer.make_fermi_net(
        nspins,
        charges,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        states=cfg.system.states,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get('jastrow', 'default'),
        bias_orbitals=cfg.network.bias_orbitals,
        rescale_inputs=cfg.network.get('rescale_inputs', False),
        complex_output=use_complex,
        **cfg.network.psiformer,
    )
  key, subkey = jax.random.split(key)
  params = network.init(subkey)
  params = kfac_jax.utils.replicate_all_local_devices(params)
  signed_network = network.apply
  # Often just need log|psi(x)|.
  if cfg.system.get('states', 0):
    if cfg.optim.objective == 'vmc_overlap':
      logabs_network = networks.make_state_trace(signed_network,
                                                 cfg.system.states)
    else:
      logabs_network = utils.select_output(
          networks.make_total_ansatz(signed_network,
                                     cfg.system.get('states', 0),
                                     complex_output=use_complex), 1)
  else:
    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
  batch_network = jax.vmap(
      logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
  )  # batched network

  # Exclusively when computing the gradient wrt the energy for complex
  # wavefunctions, it is necessary to have log(psi) rather than log(|psi|).
  # This is unused if the wavefunction is real-valued.
  if cfg.system.get('states', 0):
    if cfg.optim.objective == 'vmc_overlap':
      # In the case of a penalty method, we actually need all outputs
      # to compute the gradient
      log_network_for_loss = networks.make_state_matrix(signed_network,
                                                        cfg.system.states)
      def log_network(*args, **kwargs):
        phase, mag = log_network_for_loss(*args, **kwargs)
        return mag + 1.j * phase
    else:
      def log_network(*args, **kwargs):
        if not use_complex:
          raise ValueError('This function should never be used if the '
                           'wavefunction is real-valued.')
        meta_net = networks.make_total_ansatz(signed_network,
                                              cfg.system.get('states', 0),
                                              complex_output=True)
        phase, mag = meta_net(*args, **kwargs)
        return mag + 1.j * phase
  else:
    def log_network(*args, **kwargs):
      if not use_complex:
        raise ValueError('This function should never be used if the '
                         'wavefunction is real-valued.')
      phase, mag = signed_network(*args, **kwargs)
      return mag + 1.j * phase

  # Set up checkpointing and restore params/data if necessary
  # Mirror behaviour of checkpoints in TF FermiNet.
  # Checkpoints are saved to save_path.
  # When restoring, we first check for a checkpoint in save_path. If none are
  # found, then we check in restore_path.  This enables calculations to be
  # started from a previous calculation but then resume from their own
  # checkpoints in the event of pre-emption.

  ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
  ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)

  ckpt_restore_filename = (
      checkpoint.find_last_checkpoint(ckpt_save_path) or
      checkpoint.find_last_checkpoint(ckpt_restore_path))

  if ckpt_restore_filename:
    (t_init,
     data,
     params,
     opt_state_ckpt,
     mcmc_width_ckpt,
     density_state_ckpt) = checkpoint.restore(
         ckpt_restore_filename, host_batch_size)
  else:
    logging.info('No checkpoint found. Training new model.')
    key, subkey = jax.random.split(key)
    # make sure data on each host is initialized differently
    subkey = jax.random.fold_in(subkey, jax.process_index())
    # create electron state (position and spin)
    pos, spins = init_electrons(
        subkey,
        cfg.system.molecule,
        cfg.system.electrons,
        batch_size=total_host_batch_size,
        init_width=cfg.mcmc.init_width,
        core_electrons=core_electrons,
    )
    # For excited states, each device has a batch of walkers, where each walker
    # is nstates * nelectrons. The vmap over nstates is handled in the function
    # created in make_total_ansatz
    pos = jnp.reshape(pos, data_shape + (-1,))
    pos = kfac_jax.utils.broadcast_all_local_devices(pos)
    spins = jnp.reshape(spins, data_shape + (-1,))
    spins = kfac_jax.utils.broadcast_all_local_devices(spins)
    data = networks.FermiNetData(
        positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges
    )

    t_init = 0
    opt_state_ckpt = None
    mcmc_width_ckpt = None
    density_state_ckpt = None

  # Set up logging and observables
  train_schema = ['step', 'energy', 'ewmean', 'ewvar', 'pmove']

  if cfg.system.states:
    energy_matrix_file = open(
        os.path.join(ckpt_save_path, 'energy_matrix.npy'), 'ab')

  observable_fns = {}
  observable_states = {}  # only relevant for density matrix
  if cfg.observables.s2:
    observable_fns['s2'] = observables.make_s2(
        signed_network,
        nspins,
        states=cfg.system.states)
    observable_states['s2'] = None
    train_schema += ['s2']
    if cfg.system.states:
      s2_matrix_file = open(
          os.path.join(ckpt_save_path, 's2_matrix.npy'), 'ab')
  if cfg.observables.dipole:
    observable_fns['dipole'] = observables.make_dipole(
        signed_network,
        states=cfg.system.states)
    observable_states['dipole'] = None
    train_schema += ['mu_x', 'mu_y', 'mu_z']
    if cfg.system.states:
      dipole_matrix_file = open(
          os.path.join(ckpt_save_path, 'dipole_matrix.npy'), 'ab')
  # Do this *before* creating density matrix function, as that is a special case
  observable_fns = observables.make_observable_fns(observable_fns)

  if cfg.observables.density:
    (observable_states['density'],
     density_update,
     observable_fns['density']) = observables.make_density_matrix(
         signed_network, data.positions, cfg, density_state_ckpt)
    # Because the density matrix can be quite large, even without excited
    # states, we always save it directly to .npy file instead of writing to CSV
    density_matrix_file = open(
        os.path.join(ckpt_save_path, 'density_matrix.npy'), 'ab')
    # custom pmaping just for density matrix function
    pmap_density_axes = observables.DensityState(t=None,
                                                 positions=0,
                                                 probabilities=0,
                                                 move_width=0,
                                                 pmove=None,
                                                 mo_coeff=None)
    pmap_fn = constants.pmap(observable_fns['density'],
                             in_axes=(0, 0, pmap_density_axes))
    observable_fns['density'] = lambda *a, **kw: pmap_fn(*a, **kw).mean(0)

  # Initialisation done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

  # Pretraining to match Hartree-Fock

  if (
      t_init == 0
      and cfg.pretrain.method == 'hf'
      and cfg.pretrain.iterations > 0
  ):
    pretrain_spins = spins[0, 0]
    batch_orbitals = jax.vmap(
        network.orbitals, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    params, data.positions = pretrain.pretrain_hartree_fock(
        params=params,
        positions=data.positions,
        spins=pretrain_spins,
        atoms=data.atoms,
        charges=data.charges,
        batch_network=batch_network,
        batch_orbitals=batch_orbitals,
        network_options=network.options,
        sharded_key=subkeys,
        electrons=cfg.system.electrons,
        scf_approx=hartree_fock,
        iterations=cfg.pretrain.iterations,
        batch_size=device_batch_size,
        scf_fraction=cfg.pretrain.get('scf_fraction', 0.0),
        states=cfg.system.states,
    )

  # Main training

  # Construct MCMC step
  atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
  mcmc_step = mcmc.make_mcmc_step(
      batch_network,
      device_batch_size,
      steps=cfg.mcmc.steps,
      atoms=atoms_to_mcmc,
      blocks=cfg.mcmc.blocks * num_states,
  )
  # Construct loss and optimizer
  laplacian_method = cfg.optim.get('laplacian', 'default')
  if cfg.system.make_local_energy_fn:
    if laplacian_method != 'default':
      raise NotImplementedError(f'Laplacian method {laplacian_method}'
                                'not yet supported by custom local energy fns.')
    if cfg.optim.objective == 'vmc_overlap':
      raise NotImplementedError('Overlap penalty not yet supported for custom'
                                'local energy fns.')
    local_energy_module, local_energy_fn = (
        cfg.system.make_local_energy_fn.rsplit('.', maxsplit=1))
    local_energy_module = importlib.import_module(local_energy_module)
    make_local_energy = getattr(local_energy_module, local_energy_fn)  # type: hamiltonian.MakeLocalEnergy
    local_energy_fn = make_local_energy(
        f=signed_network,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        states=cfg.system.get('states', 0),
        **cfg.system.make_local_energy_kwargs)
  else:
    pp_symbols = cfg.system.get('pp', {'symbols': None}).get('symbols')
    local_energy_fn = hamiltonian.local_energy(
        f=signed_network,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        complex_output=use_complex,
        laplacian_method=laplacian_method,
        states=cfg.system.get('states', 0),
        state_specific=(cfg.optim.objective == 'vmc_overlap'),
        pp_type=cfg.system.get('pp', {'type': 'ccecp'}).get('type'),
        pp_symbols=pp_symbols if cfg.system.get('use_pp') else None)

  if cfg.optim.get('spin_energy', 0.0) > 0.0:
    # Minimize <H + c * S^2> instead of just <H>
    # Create a new local_energy function that takes the weighted sum of
    # the local energy and the local spin magnitude.
    local_s2_fn = observables.make_s2(
        signed_network,
        nspins=nspins,
        states=cfg.system.states)
    def local_energy_and_s2_fn(params, keys, data):
      local_energy, aux_data = local_energy_fn(params, keys, data)
      s2 = local_s2_fn(params, data, None)
      weight = cfg.optim.get('spin_energy', 0.0)
      if cfg.system.states:
        aux_data = aux_data + weight * s2
        local_energy_and_s2 = local_energy + weight * jnp.trace(s2)
      else:
        local_energy_and_s2 = local_energy + weight * s2
      return local_energy_and_s2, aux_data
    local_energy = local_energy_and_s2_fn
  else:
    local_energy = local_energy_fn

  if cfg.optim.objective == 'vmc':
    evaluate_loss = qmc_loss_functions.make_loss(
        log_network if use_complex else logabs_network,
        local_energy,
        clip_local_energy=cfg.optim.clip_local_energy,
        clip_from_median=cfg.optim.clip_median,
        center_at_clipped_energy=cfg.optim.center_at_clip,
        complex_output=use_complex,
        max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0),
    )
  elif cfg.optim.objective == 'wqmc':
    evaluate_loss = qmc_loss_functions.make_wqmc_loss(
        log_network if use_complex else logabs_network,
        local_energy,
        clip_local_energy=cfg.optim.clip_local_energy,
        clip_from_median=cfg.optim.clip_median,
        center_at_clipped_energy=cfg.optim.center_at_clip,
        complex_output=use_complex,
        max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0),
        vmc_weight=cfg.optim.get('vmc_weight', 1.0)
    )
  elif cfg.optim.objective == 'vmc_overlap':
    if not cfg.system.states:
      raise ValueError('Overlap penalty only works with excited states')
    if cfg.optim.overlap.weights is None:
      overlap_weight = tuple([1./(1.+x) for x in range(cfg.system.states)])
      overlap_weight = tuple([x/sum(overlap_weight) for x in overlap_weight])
    else:
      assert len(cfg.optim.overlap.weights) == cfg.system.states
      overlap_weight = cfg.optim.overlap.weights
    evaluate_loss = qmc_loss_functions.make_energy_overlap_loss(
        log_network_for_loss,
        local_energy,
        clip_local_energy=cfg.optim.clip_local_energy,
        clip_from_median=cfg.optim.clip_median,
        center_at_clipped_energy=cfg.optim.center_at_clip,
        overlap_penalty=cfg.optim.overlap.penalty,
        overlap_weight=overlap_weight,
        complex_output=cfg.network.get('complex', False),
        max_vmap_batch_size=cfg.optim.get('max_vmap_batch_size', 0))
  else:
    raise ValueError(f'Not a recognized objective: {cfg.optim.objective}')

  # Compute the learning rate
  def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
    return cfg.optim.lr.rate * jnp.power(
        (1.0 / (1.0 + (t_/cfg.optim.lr.delay))), cfg.optim.lr.decay)

  # Construct and setup optimizer
  if cfg.optim.optimizer == 'none':
    optimizer = None
  elif cfg.optim.optimizer == 'adam':
    optimizer = optax.chain(
        optax.scale_by_adam(**cfg.optim.adam),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))
  elif cfg.optim.optimizer == 'lamb':
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(eps=1e-7),
        optax.scale_by_trust_ratio(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1))
  elif cfg.optim.optimizer == 'kfac':
    # Differentiate wrt parameters (argument 0)
    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
    optimizer = kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=cfg.optim.kfac.l2_reg,
        norm_constraint=cfg.optim.kfac.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=cfg.optim.kfac.cov_ema_decay,
        inverse_update_period=cfg.optim.kfac.invert_every,
        min_damping=cfg.optim.kfac.min_damping,
        num_burnin_steps=0,
        register_only_generic=cfg.optim.kfac.register_only_generic,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
        ),
        # debug=True
    )
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    opt_state = optimizer.init(params, subkeys, data)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
  else:
    raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')

  if not optimizer:
    opt_state = None
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_loss_step(evaluate_loss))
  elif isinstance(optimizer, optax.GradientTransformation):
    # optax/optax-compatible optimizer (ADAM, LAMB, ...)
    opt_state = jax.pmap(optimizer.init)(params)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
        reset_if_nan=cfg.optim.reset_if_nan)
  elif isinstance(optimizer, kfac_jax.Optimizer):
    step = make_kfac_training_step(
        mcmc_step=mcmc_step,
        damping=cfg.optim.kfac.damping,
        optimizer=optimizer,
        reset_if_nan=cfg.optim.reset_if_nan)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer}')

  if mcmc_width_ckpt is not None:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
  else:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.move_width))
  pmoves = np.zeros(cfg.mcmc.adapt_frequency)

  if t_init == 0:
    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

    burn_in_step = make_training_step(
        mcmc_step=mcmc_step, optimizer_step=null_update)

    for t in range(cfg.mcmc.burn_in):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, *_ = burn_in_step(
          data,
          params,
          state=None,
          key=subkeys,
          mcmc_width=mcmc_width)
    logging.info('Completed burn-in MCMC steps')
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    ptotal_energy = constants.pmap(evaluate_loss)
    initial_energy, _ = ptotal_energy(params, subkeys, data)
    logging.info('Initial energy: %03.4f E_h', initial_energy[0])

  time_of_last_ckpt = time.time()
  weighted_stats = None

  if cfg.optim.optimizer == 'none' and opt_state_ckpt is not None:
    # If opt_state_ckpt is None, then we're restarting from a previous inference
    # run (most likely due to preemption) and so should continue from the last
    # iteration in the checkpoint. Otherwise, starting an inference run from a
    # training run.
    logging.info('No optimizer provided. Assuming inference run.')
    logging.info('Setting initial iteration to 0.')
    t_init = 0

    # Excited states inference only: rescale each state to be roughly
    # comparable, to avoid large outlier values in the local energy matrix.
    # This is not a factor in training, as the outliers are only off-diagonal.
    # This only becomes a significant factor for systems with >25 electrons.
    if cfg.system.states > 0 and 'state_scale' not in params:
      state_matrix = utils.select_output(
          networks.make_state_matrix(signed_network,
                                     cfg.system.states), 1)
      batch_state_matrix = jax.vmap(state_matrix, (None, 0, 0, 0, 0))
      pmap_state_matrix = constants.pmap(batch_state_matrix)
      log_psi_vals = pmap_state_matrix(
          params, data.positions, data.spins, data.atoms, data.charges)
      state_scale = np.mean(log_psi_vals, axis=[0, 1, 2])
      state_scale = jax.experimental.multihost_utils.broadcast_one_to_all(
          state_scale)
      state_scale = np.tile(state_scale[None], [jax.local_device_count(), 1])
      if isinstance(params, dict):  # Always true, but prevents type errors
        params['state_scale'] = -state_scale

  if writer_manager is None:
    writer_manager = writers.Writer(
        name='train_stats',
        schema=train_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False)
  with writer_manager as writer:
    # Main training loop
    num_resets = 0  # used if reset_if_nan is true
    for t in range(t_init, cfg.optim.iterations):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, opt_state, loss, aux_data, pmove = step(
          data,
          params,
          opt_state,
          subkeys,
          mcmc_width)

      # due to pmean, loss, and pmove should be the same across
      # devices.
      loss = loss[0]
      # per batch variance isn't informative. Use weighted mean and variance
      # instead.
      weighted_stats = statistics.exponentialy_weighted_stats(
          alpha=0.1, observation=loss, previous_stats=weighted_stats)
      pmove = pmove[0]

      # Update observables
      observable_data = {
          key: fn(params, data, observable_states[key])
          for key, fn in observable_fns.items()
      }
      if cfg.observables.density:
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        observable_states['density'] = density_update(
            subkeys, params, data, observable_states['density'])

      # Update MCMC move width
      mcmc_width, pmoves = mcmc.update_mcmc_width(
          t, mcmc_width, cfg.mcmc.adapt_frequency, pmove, pmoves)

      if cfg.debug.check_nan:
        tree = {'params': params, 'loss': loss}
        if cfg.optim.optimizer != 'none':
          tree['optim'] = opt_state
        try:
          chex.assert_tree_all_finite(tree)
          num_resets = 0  # Reset counter if check passes
        except AssertionError as e:
          if cfg.optim.reset_if_nan:  # Allow a certain number of NaNs
            num_resets += 1
            if num_resets > 100:
              raise e
          else:
            raise e

      # Logging
      if t % cfg.log.stats_frequency == 0:
        logging_str = ('Step %05d: '
                       '%03.4f E_h, exp. variance=%03.4f E_h^2, pmove=%0.2f')
        logging_args = t, loss, weighted_stats.variance, pmove
        writer_kwargs = {
            'step': t,
            'energy': np.asarray(loss),
            'ewmean': np.asarray(weighted_stats.mean),
            'ewvar': np.asarray(weighted_stats.variance),
            'pmove': np.asarray(pmove),
        }
        for key in observable_data:
          obs_data = observable_data[key]
          if cfg.system.states:
            obs_data = np.trace(obs_data, axis1=-1, axis2=-2)
          if key == 'dipole':
            writer_kwargs['mu_x'] = obs_data[0]
            writer_kwargs['mu_y'] = obs_data[1]
            writer_kwargs['mu_z'] = obs_data[2]
          elif key == 'density':
            pass
          elif key == 's2':
            writer_kwargs[key] = obs_data
            logging_str += ', <S^2>=%03.4f'
            logging_args += obs_data,
        logging.info(logging_str, *logging_args)
        writer.write(t, **writer_kwargs)

      # Log data about observables too big to fit in a CSV
      if cfg.system.states:
        energy_matrix = aux_data.local_energy_mat
        energy_matrix = np.nanmean(np.nanmean(energy_matrix, axis=0), axis=0)
        np.save(energy_matrix_file, energy_matrix)
        if cfg.observables.s2:
          np.save(s2_matrix_file, observable_data['s2'])
        if cfg.observables.dipole:
          np.save(dipole_matrix_file, observable_data['dipole'])
      if cfg.observables.density:
        np.save(density_matrix_file, observable_data['density'])

      # Checkpointing
      if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60:
        checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
        time_of_last_ckpt = time.time()

    # Shut down logging at end
    if cfg.system.states:
      energy_matrix_file.close()
      if cfg.observables.s2:
        s2_matrix_file.close()
      if cfg.observables.dipole:
        dipole_matrix_file.close()
    if cfg.observables.density:
      density_matrix_file.close()
