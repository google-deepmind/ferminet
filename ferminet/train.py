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
import time
from typing import Sequence

from absl import logging
import chex
from ferminet import checkpoint
from ferminet import constants
from ferminet import hamiltonian
from ferminet import jax_utils
from ferminet import mcmc
from ferminet import networks
from ferminet import pretrain
from ferminet.utils import system
from ferminet.utils import writers
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax


def init_electrons(
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
) -> jnp.ndarray:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.

  Returns:
    array of (batch_size, nalpha*nbeta*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3).
  """
  if sum(atom.charge for atom in molecule) != sum(electrons):
    if len(molecule) == 1:
      atomic_spin_configs = [electrons]
    else:
      raise NotImplementedError('No initialization policy yet '
                                'exists for charged molecules.')
  else:
    atomic_spin_configs = [
        (atom.element.nalpha, atom.element.nbeta) for atom in molecule
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
  return (
      electron_positions +
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size)))


def make_loss(network, batch_network, atoms, charges, clip_local_energy=0.0):
  """Creates the loss function, including custom gradients.

  Args:
    network: function, signature (params, data), which evaluates the log of
      the wavefunction (square root of the log probability distribution) at the
      single MCMC configuration in data given the network parameters.
    batch_network: as for network but data is a batch of MCMC configurations.
    atoms: array of (natoms, ndim) specifying the positions of the nuclei.
    charges: array of (natoms) specifying the nuclear charges.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.

  Returns:
    Callable with signature (params, data) which evaluates the energy of the
    network given the parameters and the (batched) MCMC configurations in data.
  """
  el_fun = hamiltonian.local_energy(network, atoms, charges)
  batch_local_energy = jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)

  @jax.custom_jvp
  def total_energy(params, data):
    """Evaluates the total energy of the network for a batch of configurations.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.

    Returns:
      Mean total energy across the batch, over all devices if inside a pmap.
    """
    e_l = batch_local_energy(params, data)
    loss = jax.lax.pmean(jnp.mean(e_l), axis_name=constants.PMAP_AXIS_NAME)
    return loss

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    e_l = batch_local_energy(*primals)
    loss = jax.lax.pmean(jnp.mean(e_l), axis_name=constants.PMAP_AXIS_NAME)

    if clip_local_energy > 0.0:
      # Try centering the window around the median instead of the mean?
      tv = jnp.mean(jnp.abs(e_l - loss))
      tv = jax.lax.pmean(tv, axis_name=constants.PMAP_AXIS_NAME)
      diff = jnp.clip(e_l,
                      loss - clip_local_energy*tv,
                      loss + clip_local_energy*tv) - loss
    else:
      diff = e_l - loss

    _, psi_tangent = jax.jvp(batch_network, primals, tangents)
    return loss, jnp.dot(psi_tangent, diff)

  return total_energy


def make_training_step(mcmc_step, val_and_grad, opt_update):
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    val_and_grad: Callable f(params, data) which evaluates the loss and
      gradients given network parameters and MCMC configurations.
    opt_update: Callable f(t, gradients, params, state) which updates the
      network parameters according to an optimizer policy and returns the
      updated network parameters and optimization state.

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the step docstring for details.
  """
  @functools.partial(constants.pmap, donate_argnums=(1, 2, 3, 4))
  def step(t, data, params, state, key, mcmc_width):
    """A full update iteration (except for KFAC): MCMC steps + optimization.

    Args:
      t: training step iteration.
      data: batch of MCMC configurations.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        pmove: probability that a proposed MCMC move was accepted.
    """
    # MCMC loop
    # Should this be created outside the function?
    data, pmove = mcmc_step(params, data, key, mcmc_width)

    # Optimization step
    loss, search_direction = val_and_grad(params, data)
    search_direction = jax.lax.pmean(
        search_direction, axis_name=constants.PMAP_AXIS_NAME)
    state, params = opt_update(t, search_direction, params, state)
    return data, params, state, loss, pmove
  return step


def pyscf_to_molecule(cfg: ml_collections.ConfigDict):
  """Converts the PySCF 'Molecule' in the config to the internal representation.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details. Must have the system.pyscf_mol set.

  Returns:
    cfg: ConfigDict matching the input with system.molecule, system.electrons
      and pretrain.basis fields set from the information in the system.pyscf_mol
      field.

  Raises:
    ValueError: if the system.pyscf_mol field is not set in the cfg.
  """
  if not cfg.system.pyscf_mol:
    raise ValueError('You must set system.pyscf_mol in your cfg')
  cfg.system.pyscf_mol.build()
  cfg.system.electrons = cfg.system.pyscf_mol.nelec
  cfg.system.molecule = [system.Atom(cfg.system.pyscf_mol.atom_symbol(i),
                                     cfg.system.pyscf_mol.atom_coords()[i])
                         for i in range(cfg.system.pyscf_mol.natm)]
  cfg.pretrain.basis = cfg.system.pyscf_mol.basis
  return cfg


def train(cfg: ml_collections.ConfigDict):
  """Runs training loop for QMC.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
  # Device logging
  num_devices = jax.device_count()
  logging.info('Starting QMC with %i XLA devices', num_devices)
  if cfg.batch_size % num_devices != 0:
    raise ValueError('Batch size must be divisible by number of devices, '
                     'got batch size {} for {} devices.'.format(
                         cfg.batch_size, num_devices))
  if cfg.system.ndim != 3:
    # The network (at least the input feature construction) and initial MCMC
    # molecule configuration (via system.Atom) assume 3D systems. This can be
    # lifted with a little work.
    raise ValueError('Only 3D systems are currently supported.')
  data_shape = (num_devices, cfg.batch_size // num_devices)

  # Check if mol is a pyscf molecule and convert to internal representation
  if cfg.system.pyscf_mol:
    cfg = pyscf_to_molecule(cfg)

  # Convert mol config into array of atomic positions and charges
  atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
  charges = jnp.array([atom.charge for atom in cfg.system.molecule])
  spins = cfg.system.electrons

  if cfg.debug.deterministic:
    seed = 23
  else:
    seed = int(1e6 * time.time())
  key = jax.random.PRNGKey(seed)

  # Create parameters, network, and vmaped/pmaped derivations

  if cfg.pretrain.method == 'direct_init' or (
      cfg.pretrain.method == 'hf' and cfg.pretrain.iterations > 0):
    if cfg.system.pyscf_mol:
      hartree_fock = pretrain.get_hf(
          pyscf_mol=cfg.system.pyscf_mol, restricted=False)
    else:
      hartree_fock = pretrain.get_hf(
          cfg.system.molecule, cfg.system.electrons,
          basis=cfg.pretrain.basis, restricted=False)

  hf_solution = hartree_fock if cfg.pretrain.method == 'direct_init' else None
  network_init, network = networks.make_fermi_net(
      atoms, spins, charges,
      envelope_type=cfg.network.envelope_type,
      bias_orbitals=cfg.network.bias_orbitals,
      use_last_layer=cfg.network.use_last_layer,
      hf_solution=hf_solution,
      full_det=cfg.network.full_det,
      **cfg.network.detnet)
  key, subkey = jax.random.split(key)
  params = network_init(subkey)
  params = jax_utils.replicate(params)
  batch_network = jax.vmap(network, (None, 0), 0)  # batched network

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
    t_init, data, params, opt_state_ckpt, mcmc_width_ckpt = checkpoint.restore(
        ckpt_restore_filename, cfg.batch_size)
  else:
    logging.info('No checkpoint found. Training new model.')
    key, subkey = jax.random.split(key)
    data = init_electrons(subkey, cfg.system.molecule, cfg.system.electrons,
                          cfg.batch_size)
    data = jnp.reshape(data, data_shape + data.shape[1:])
    data = jax_utils.broadcast(data)
    t_init = 0
    opt_state_ckpt = None
    mcmc_width_ckpt = None

  # Set up logging
  train_schema = ['step', 'energy', 'pmove']

  # Initialisation done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  key, subkeys = jax.random.split(key, num_devices+1)
  subkeys = jnp.stack(subkeys)
  # Handle num_devices=1 case by explicitly broadcasting from an array of shape
  # (2,) to an array of (num_devices, 2). This is a no-op for num_devices > 1.
  subkeys = jnp.broadcast_to(subkeys, (num_devices, 2))
  sharded_key = jax_utils.broadcast(subkeys)

  # Pretraining to match Hartree-Fock

  if (t_init == 0 and cfg.pretrain.method == 'hf' and
      cfg.pretrain.iterations > 0):
    sharded_key, subkeys = jax_utils.p_split(sharded_key)
    params, data = pretrain.pretrain_hartree_fock(
        params,
        data,
        batch_network,
        subkeys,
        cfg.system.molecule,
        cfg.system.electrons,
        scf_approx=hartree_fock,
        envelope_type=cfg.network.envelope_type,
        full_det=cfg.network.full_det,
        iterations=cfg.pretrain.iterations)

  # Main training

  # Construct MCMC step
  atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
  mcmc_step = mcmc.make_mcmc_step(
      batch_network,
      cfg.batch_size // num_devices,
      steps=cfg.mcmc.steps,
      atoms=atoms_to_mcmc,
      one_electron_moves=cfg.mcmc.one_electron)
  # Construct loss and optimizer
  total_energy = make_loss(network, batch_network, atoms, charges,
                           clip_local_energy=cfg.optim.clip_el)
  # Compute the learning rate
  def learning_rate_schedule(t):
    return cfg.optim.lr.rate * jnp.power(
        (1.0 / (1.0 + (t/cfg.optim.lr.delay))), cfg.optim.lr.decay)
  # Differentiate wrt parameters (argument 0)
  val_and_grad = jax.value_and_grad(total_energy, argnums=0, has_aux=False)
  if cfg.optim.optimizer == 'adam':
    optimizer = optax.chain(
        optax.scale_by_adam(**cfg.optim.adam),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))
  elif cfg.optim.optimizer == 'none':
    total_energy = constants.pmap(total_energy)
    opt_state = None
  else:
    raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')

  if cfg.optim.optimizer != 'none':
    opt_state = jax.pmap(optimizer.init)(params)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    def opt_update(t, grad, params, opt_state):
      del t
      updates, opt_state = optimizer.update(grad, opt_state, params)  # pytype: disable=wrong-arg-count,attribute-error
      params = optax.apply_updates(params, updates)
      return opt_state, params
    step = make_training_step(mcmc_step, val_and_grad, opt_update)
  # Only the pmapped MCMC step is needed after this point
  mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)

  # The actual training loop

  mcmc_width = (mcmc_width_ckpt if mcmc_width_ckpt is not None
                else jax_utils.replicate(jnp.asarray(cfg.mcmc.move_width)))
  pmoves = np.zeros(cfg.mcmc.adapt_frequency)
  shared_t = jax_utils.replicate(jnp.zeros([]))

  if t_init == 0:
    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)
    for t in range(cfg.mcmc.burn_in):
      sharded_key, subkeys = jax_utils.p_split(sharded_key)
      data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
    logging.info('Completed burn-in MCMC steps')
    logging.info('Initial energy: %03.4f E_h',
                 constants.pmap(total_energy)(params, data))

  time_of_last_ckpt = time.time()

  if cfg.optim.optimizer == 'none' and opt_state_ckpt is not None:
    # If opt_state_ckpt is None, then we're restarting from a previous inference
    # run (most likely due to preemption) and so should continue from the last
    # iteration in the checkpoint. Otherwise, starting an inference run from a
    # training run.
    logging.info('No optimizer provided. Assuming inference run.')
    logging.info('Setting initial iteration to 0.')
    t_init = 0

  with writers.Writer(
      name='train_stats',
      schema=train_schema,
      directory=ckpt_save_path,
      iteration_key=None,
      log=False) as writer:
    for t in range(t_init, cfg.optim.iterations):
      sharded_key, subkeys = jax_utils.p_split(sharded_key)
      if cfg.optim.optimizer == 'none':
        data, pmove = mcmc_step(params, data, subkeys, mcmc_width)
        loss = total_energy(params, data)
      else:
        data, params, opt_state, loss, pmove = step(
            shared_t,
            data,
            params,
            opt_state,
            subkeys,
            mcmc_width)
        shared_t = shared_t + 1

      # due to pmean, both loss and pmove should be the same across devices.
      loss = loss[0]
      pmove = pmove[0]

      # Update MCMC move width
      if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
        if np.mean(pmoves) > 0.55:
          mcmc_width *= 1.1
        if np.mean(pmoves) < 0.5:
          mcmc_width /= 1.1
        pmoves[:] = 0
      pmoves[t%cfg.mcmc.adapt_frequency] = pmove

      if cfg.debug.check_nan:
        tree = {'params': params, 'loss': loss}
        if cfg.optim.optimizer != 'none':
          tree['optim'] = opt_state
        chex.assert_tree_all_finite(tree)

      # Logging
      if t % cfg.log.stats_frequency == 0:
        logging.info('Step %05d: %03.4f E_h, pmove=%0.2f', t, loss, pmove)
        writer.write(t, step=t, energy=loss._npy_value, pmove=pmove._npy_value)  # pylint: disable=protected-access

      # Checkpointing
      if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60:
        checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
        time_of_last_ckpt = time.time()
