# Lint as: python3
# Copyright 2018 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Learn ground state wavefunctions for molecular systems using VMC."""

import copy
import os
from typing import Any, Mapping, Optional, Sequence, Tuple

from absl import logging
import attr
from ferminet import hamiltonian
from ferminet import mcmc
from ferminet import mean_corrected_kfac_opt
from ferminet import networks
from ferminet import qmc
from ferminet import scf
from ferminet.utils import elements
from ferminet.utils import system
import numpy as np
import tensorflow.compat.v1 as tf


def _validate_directory(obj, attribute, value):
  """Validates value is a directory."""
  del obj
  if value and not os.path.isdir(value):
    raise ValueError(f'{attribute.name} is not a directory')


@attr.s(auto_attribs=True)
class LoggingConfig:
  """Logging information for Fermi Nets.

  Attributes:
    result_path: directory to use for saving model parameters and calculations
      results. Created if does not exist.
    save_frequency: frequency (in minutes) at which parameters are saved.
    restore_path: directory to use for restoring model parameters.
    stats_frequency: frequency (in iterations) at which statistics (via stats
      hooks) are updated and stored.
    replicas: the number of replicas used during training. Will be set
      automatically.
    walkers: If true, log walkers at every step.
    wavefunction: If true, log wavefunction at every step.
    local_energy: If true, log local energy at every step.
    config: dictionary of additional information about the calculation setup.
      Reported along with calculation statistics.
  """
  result_path: str = '.'
  save_frequency: float = 10
  restore_path: str = attr.ib(default=None, validator=_validate_directory)
  stats_frequency: int = 1
  replicas: int = 1
  walkers: bool = False
  wavefunction: bool = False
  local_energy: bool = False
  config: Mapping[str, Any] = attr.ib(converter=dict,
                                      default=attr.Factory(dict))


@attr.s(auto_attribs=True)
class MCMCConfig:
  """Markov Chain Monte Carlo configuration for Fermi Nets.

  Attributes:
    burn_in: Number of burn in steps after pretraining.
    steps: 'Number of MCMC steps to make between network updates.
    init_width: Width of (atom-centred) Gaussians used to generate initial
      electron configurations.
    move_width: Width of Gaussian used for random moves.
    init_means: Iterable of 3*nelectrons giving the mean initial position of
      each electron. Configurations are drawn using Gaussians of width
      init_width at each 3D position. Alpha electrons are listed before beta
      electrons. If empty, electrons are assigned to atoms based upon the
      isolated atom spin configuration. Expert use only.
  """
  burn_in: int = 100
  steps: int = 10
  init_width: float = 0.8
  move_width: float = 0.02
  init_means: Optional[Sequence[float]] = None


@attr.s(auto_attribs=True)
class PretrainConfig:
  """Hartree-Fock pretraining algorithm configuration for Fermi Nets.

  Attributes:
    iterations: Number of iterations for which to pretrain the network to match
      Hartree-Fock orbitals.
    basis: Basis set used to run Hartree-Fock calculation in PySCF.
  """
  iterations: int = 1000
  basis: str = 'sto-3g'


@attr.s(auto_attribs=True)
class OptimConfig:
  """Optimization configuration for Fermi Nets.

  Attributes:
    iterations: Number of iterations')
    clip_el: If not none, scale at which to clip local energy.
    learning_rate: learning rate.
    learning_rate_decay: exponent of learning rate decay.
    learning_rate_delay: scale of the rate decay.
    use_kfac: Use the K-FAC optimizer if true, ADAM optimizer otherwise.
    check_loss: Apply gradient update only if the loss is not NaN.  If true,
      training could be slightly slower but the checkpoint written out when a
      NaN is detected will be with the network weights which led to the NaN.
    deterministic: CPU only mode that also enforces determinism. Will run
      *significantly* slower if used.
  """
  iterations: int = 1000000
  learning_rate: float = 1.e-4
  learning_rate_decay: float = 1.0
  learning_rate_delay: float = 10000.0
  clip_el: float = 5.0
  use_kfac: bool = True
  check_loss: bool = False
  deterministic: bool = False


@attr.s(auto_attribs=True)
class KfacConfig:
  """K-FAC configuration - see docs at https://github.com/tensorflow/kfac/."""
  invert_every: int = 1
  cov_update_every: int = 1
  damping: float = 0.001
  cov_ema_decay: float = 0.95
  momentum: float = 0.0
  momentum_type: str = attr.ib(
      default='regular',
      validator=attr.validators.in_(
          ['regular', 'adam', 'qmodel', 'qmodel_fixedmu']))
  adapt_damping: bool = False
  damping_adaptation_decay: float = 0.9
  damping_adaptation_interval: int = 5
  min_damping: float = 1.e-5
  norm_constraint: float = 0.001


@attr.s(auto_attribs=True)
class NetworkConfig:
  """Network configuration for Fermi Net.

  Attributes:
    architecture: The choice of architecture to run the calculation with. Either
      "ferminet" or "slater" for the Fermi Net and standard Slater determinant
      respectively.
    hidden_units: Number of hidden units in each layer of the network. If
      the Fermi Net with one- and two-electron streams is used, a tuple is
      provided for each layer, with the first element giving the number of
      hidden units in the one-electron stream and the second element giving the
      number of units in the two-electron stream. Otherwise, each layer is
      represented by a single integer.
    determinants: Number of determinants to use.
    r12_en_features: Include r12/distance features between electrons and nuclei.
      Highly recommended.
    r12_ee_features: Include r12/distance features between pairs of electrons.
      Highly recommended.
    pos_ee_features: Include electron-electron position features. Highly
      recommended.
    use_envelope: Include multiplicative exponentially-decaying envelopes on
      each orbital. Calculations will not converge if set to False.
    backflow: Include backflow transformation in input coordinates. '
      Only for use if network_architecture == "slater". Implies build_backflow
      is also True.
    build_backflow: Create backflow weights but do not include backflow
      coordinate transformation in the netwrok. Use to train a Slater-Jastrow
      architecture and then train a Slater-Jastrow-Backflow architecture
      based on it in a two-stage optimization process.
    residual: Use residual connections in network. Recommended.
    after_det: Number of hidden units in each layer of the neural network after
      the determinants. By default, just takes a  weighted sum of
      determinants with no nonlinearity.
    jastrow_en: Include electron-nuclear Jastrow factor. Only relevant with
      Slater-Jastrow-Backflow architectures.
    jastrow_ee: Include electron-electron Jastrow factor. Only relevant with
      Slater-Jastrow-Backflow architectures.
    jastrow_een: Include electron-electron-nuclear Jastrow factor. Only
      relevant with Slater-Jastrow-Backflow architectures.
  """
  architecture: str = attr.ib(
      default='ferminet', validator=attr.validators.in_(['ferminet', 'slater']))
  hidden_units: Sequence[Tuple[int, int]] = ((256, 32),) * 4
  determinants: int = 16
  r12_en_features: bool = True
  r12_ee_features: bool = True
  pos_ee_features: bool = True
  use_envelope: bool = True
  backflow: bool = False
  build_backflow: bool = False
  residual: bool = True
  after_det: Sequence[int] = (1,)
  jastrow_en: bool = False
  jastrow_ee: bool = False
  jastrow_een: bool = False


def assign_electrons(molecule, electrons):
  """Assigns electrons to atoms using non-interacting spin configurations.

  Args:
    molecule: List of Hamiltonian.Atom objects for each atom in the system.
    electrons: Pair of ints giving number of alpha (spin-up) and beta
      (spin-down) electrons.

  Returns:
    1D np.ndarray of length 3N containing initial mean positions of each
    electron based upon the atom positions, where N is the total number of
    electrons. The first 3*electrons[0] positions correspond to the alpha
    (spin-up) electrons and the next 3*electrons[1] to the beta (spin-down)
    electrons.

  Raises:
    RuntimeError: if a different number of electrons or different spin
    polarisation is generated.
  """
  # Assign electrons based upon unperturbed atoms and ignore impact of
  # fractional nuclear charge.
  nuclei = [int(round(atom.charge)) for atom in molecule]
  total_charge = sum(nuclei) - sum(electrons)
  # Construct a dummy iso-electronic neutral system.
  neutral_molecule = [copy.copy(atom) for atom in molecule]
  if total_charge != 0:
    logging.warning(
        'Charged system. Using heuristics to set initial electron positions')
    charge = 1 if total_charge > 0 else -1
  while total_charge != 0:
    # Poor proxy for electronegativity.
    atom_index = nuclei.index(max(nuclei) if total_charge < 0 else min(nuclei))
    atom = neutral_molecule[atom_index]
    atom.charge -= charge
    atom.atomic_number = int(round(atom.charge))
    if int(round(atom.charge)) == 0:
      neutral_molecule.pop(atom_index)
    else:
      atom.symbol = elements.ATOMIC_NUMS[atom.atomic_number].symbol
    total_charge -= charge
    nuclei = [int(round(atom.charge)) for atom in neutral_molecule]

  spin_pol = lambda electrons: electrons[0] - electrons[1]
  abs_spin_pol = abs(spin_pol(electrons))
  if len(neutral_molecule) == 1:
    elecs_atom = [electrons]
  else:
    elecs_atom = []
    spin_pol_assigned = 0
    for ion in neutral_molecule:
      # Greedily assign up and down electrons based upon the ground state spin
      # configuration of an isolated atom.
      atom_spin_pol = elements.ATOMIC_NUMS[ion.atomic_number].spin_config
      nelec = ion.atomic_number
      na = (nelec + atom_spin_pol) // 2
      nb = nelec - na
      # Attempt to keep spin polarisation as close to 0 as possible.
      if (spin_pol_assigned > 0 and
          spin_pol_assigned + atom_spin_pol > abs_spin_pol):
        elec_atom = [nb, na]
      else:
        elec_atom = [na, nb]
      spin_pol_assigned += spin_pol(elec_atom)
      elecs_atom.append(elec_atom)

  electrons_assigned = [sum(e) for e in zip(*elecs_atom)]
  spin_pol_assigned = spin_pol(electrons_assigned)
  if np.sign(spin_pol_assigned) == -np.sign(abs_spin_pol):
    # Started with the wrong guess for spin-up vs spin-down.
    elecs_atom = [e[::-1] for e in elecs_atom]
    spin_pol_assigned = -spin_pol_assigned

  if spin_pol_assigned != abs_spin_pol:
    logging.info('Spin polarisation does not match isolated atoms. '
                 'Using heuristics to set initial electron positions.')
  while spin_pol_assigned != abs_spin_pol:
    atom_spin_pols = [abs(spin_pol(e)) for e in elecs_atom]
    atom_index = atom_spin_pols.index(max(atom_spin_pols))
    elec_atom = elecs_atom[atom_index]
    if spin_pol_assigned < abs_spin_pol and elec_atom[0] <= elec_atom[1]:
      elec_atom[0] += 1
      elec_atom[1] -= 1
      spin_pol_assigned += 2
    elif spin_pol_assigned < abs_spin_pol and elec_atom[0] > elec_atom[1]:
      elec_atom[0] -= 1
      elec_atom[1] += 1
      spin_pol_assigned += 2
    elif spin_pol_assigned > abs_spin_pol and elec_atom[0] > elec_atom[1]:
      elec_atom[0] -= 1
      elec_atom[1] += 1
      spin_pol_assigned -= 2
    else:
      elec_atom[0] += 1
      elec_atom[1] -= 1
      spin_pol_assigned -= 2

  electrons_assigned = [sum(e) for e in zip(*elecs_atom)]
  if spin_pol(electrons_assigned) == -spin_pol(electrons):
    elecs_atom = [e[::-1] for e in elecs_atom]
    electrons_assigned = electrons_assigned[::-1]

  logging.info(
      'Electrons assigned %s.', ', '.join([
          '{}: {}'.format(atom.symbol, elec_atom)
          for atom, elec_atom in zip(molecule, elecs_atom)
      ]))
  if any(e != e_assign for e, e_assign in zip(electrons, electrons_assigned)):
    raise RuntimeError(
        'Assigned incorrect number of electrons ([%s instead of %s]' %
        (electrons_assigned, electrons))
  if any(min(ne) < 0 for ne in zip(*elecs_atom)):
    raise RuntimeError('Assigned negative number of electrons!')
  electron_positions = np.concatenate([
      np.tile(atom.coords, e[0])
      for atom, e in zip(neutral_molecule, elecs_atom)
  ] + [
      np.tile(atom.coords, e[1])
      for atom, e in zip(neutral_molecule, elecs_atom)
  ])
  return electron_positions


def train(molecule: Sequence[system.Atom],
          spins: Tuple[int, int],
          batch_size: int,
          network_config: Optional[NetworkConfig] = None,
          pretrain_config: Optional[PretrainConfig] = None,
          optim_config: Optional[OptimConfig] = None,
          kfac_config: Optional[KfacConfig] = None,
          mcmc_config: Optional[MCMCConfig] = None,
          logging_config: Optional[LoggingConfig] = None,
          multi_gpu: bool = False,
          double_precision: bool = False,
          graph_path: Optional[str] = None):
  """Configures and runs training loop.

  Args:
    molecule: molecule description.
    spins: pair of ints specifying number of spin-up and spin-down electrons
      respectively.
    batch_size: batch size. Also referred to as the number of Markov Chain Monte
      Carlo configurations/walkers.
    network_config: network configuration. Default settings in NetworkConfig are
      used if not specified.
    pretrain_config: pretraining configuration. Default settings in
      PretrainConfig are used if not specified.
    optim_config: optimization configuration. Default settings in OptimConfig
      are used if not specified.
    kfac_config: K-FAC configuration. Default settings in KfacConfig are used if
      not specified.
    mcmc_config: Markov Chain Monte Carlo configuration. Default settings in
      MCMCConfig are used if not specified.
    logging_config: logging and checkpoint configuration. Default settings in
      LoggingConfig are used if not specified.
    multi_gpu: Use all available GPUs. Default: use only a single GPU.
    double_precision: use tf.float64 instead of tf.float32 for all operations.
      Warning - double precision is not currently functional with K-FAC.
    graph_path: directory to save a representation of the TF graph to. Not saved

  Raises:
    RuntimeError: if mcmc_config.init_means is supplied but is of the incorrect
    length.
  """

  if not mcmc_config:
    mcmc_config = MCMCConfig()
  if not logging_config:
    logging_config = LoggingConfig()
  if not pretrain_config:
    pretrain_config = PretrainConfig()
  if not optim_config:
    optim_config = OptimConfig()
  if not kfac_config:
    kfac_config = KfacConfig()
  if not network_config:
    network_config = NetworkConfig()

  nelectrons = sum(spins)
  precision = tf.float64 if double_precision else tf.float32

  if multi_gpu:
    strategy = tf.distribute.MirroredStrategy()
  else:
    # Get the default (single-device) strategy.
    strategy = tf.distribute.get_strategy()
  if multi_gpu:
    batch_size = batch_size // strategy.num_replicas_in_sync
    logging.info('Setting per-GPU batch size to %s.', batch_size)
    logging_config.replicas = strategy.num_replicas_in_sync
  logging.info('Running on %s replicas.', strategy.num_replicas_in_sync)

  # Create a re-entrant variable scope for network.
  with tf.variable_scope('model') as model:
    pass

  with strategy.scope():
    with tf.variable_scope(model, auxiliary_name_scope=False) as model1:
      with tf.name_scope(model1.original_name_scope):
        fermi_net = networks.FermiNet(
            atoms=molecule,
            nelectrons=spins,
            slater_dets=network_config.determinants,
            hidden_units=network_config.hidden_units,
            after_det=network_config.after_det,
            architecture=network_config.architecture,
            r12_ee_features=network_config.r12_ee_features,
            r12_en_features=network_config.r12_en_features,
            pos_ee_features=network_config.pos_ee_features,
            build_backflow=network_config.build_backflow,
            use_backflow=network_config.backflow,
            jastrow_en=network_config.jastrow_en,
            jastrow_ee=network_config.jastrow_ee,
            jastrow_een=network_config.jastrow_een,
            logdet=True,
            envelope=network_config.use_envelope,
            residual=network_config.residual,
            pretrain_iterations=pretrain_config.iterations)

    scf_approx = scf.Scf(
        molecule,
        nelectrons=spins,
        restricted=False,
        basis=pretrain_config.basis)
    if pretrain_config.iterations > 0:
      scf_approx.run()

    hamiltonian_ops = hamiltonian.operators(molecule, nelectrons)
    if mcmc_config.init_means:
      if len(mcmc_config.init_means) != 3 * nelectrons:
        raise RuntimeError('Initial electron positions of incorrect shape. '
                           '({} not {})'.format(
                               len(mcmc_config.init_means), 3 * nelectrons))
      init_means = [float(x) for x in mcmc_config.init_means]
    else:
      init_means = assign_electrons(molecule, spins)

    # Build the MCMC state inside the same variable scope as the network.
    with tf.variable_scope(model, auxiliary_name_scope=False) as model1:
      with tf.name_scope(model1.original_name_scope):
        data_gen = mcmc.MCMC(
            fermi_net,
            batch_size,
            init_mu=init_means,
            init_sigma=mcmc_config.init_width,
            move_sigma=mcmc_config.move_width,
            dtype=precision)
    with tf.variable_scope('HF_data_gen'):
      hf_data_gen = mcmc.MCMC(
          scf_approx.tf_eval_slog_hartree_product,
          batch_size,
          init_mu=init_means,
          init_sigma=mcmc_config.init_width,
          move_sigma=mcmc_config.move_width,
          dtype=precision)

    with tf.name_scope('learning_rate_schedule'):
      global_step = tf.train.get_or_create_global_step()
      lr = optim_config.learning_rate * tf.pow(
          (1.0 / (1.0 + (tf.cast(global_step, tf.float32) /
                         optim_config.learning_rate_delay))),
          optim_config.learning_rate_decay)

    if optim_config.learning_rate < 1.e-10:
      logging.warning('Learning rate less than 10^-10. Not using an optimiser.')
      optim_fn = lambda _: None
      update_cached_data = None
    elif optim_config.use_kfac:
      cached_data = tf.get_variable(
          'MCMC_cache',
          initializer=tf.zeros(shape=data_gen.walkers.shape, dtype=precision),
          use_resource=True,
          trainable=False,
          dtype=precision,
      )
      if kfac_config.adapt_damping:
        update_cached_data = tf.assign(cached_data, data_gen.walkers)
      else:
        update_cached_data = None
      optim_fn = lambda layer_collection: mean_corrected_kfac_opt.MeanCorrectedKfacOpt(  # pylint: disable=g-long-lambda
          invert_every=kfac_config.invert_every,
          cov_update_every=kfac_config.cov_update_every,
          learning_rate=lr,
          norm_constraint=kfac_config.norm_constraint,
          damping=kfac_config.damping,
          cov_ema_decay=kfac_config.cov_ema_decay,
          momentum=kfac_config.momentum,
          momentum_type=kfac_config.momentum_type,
          loss_fn=lambda x: tf.nn.l2_loss(fermi_net(x)[0]),
          train_batch=data_gen.walkers,
          prev_train_batch=cached_data,
          layer_collection=layer_collection,
          batch_size=batch_size,
          adapt_damping=kfac_config.adapt_damping,
          is_chief=True,
          damping_adaptation_decay=kfac_config.damping_adaptation_decay,
          damping_adaptation_interval=kfac_config.damping_adaptation_interval,
          min_damping=kfac_config.min_damping,
          use_passed_loss=False,
          estimation_mode='exact',
      )
    else:
      adam = tf.train.AdamOptimizer(lr)
      optim_fn = lambda _: adam
      update_cached_data = None

    qmc_net = qmc.QMC(
        hamiltonian_ops,
        fermi_net,
        data_gen,
        hf_data_gen,
        clip_el=optim_config.clip_el,
        check_loss=optim_config.check_loss,
    )

  qmc_net.train(
      optim_fn,
      optim_config.iterations,
      logging_config,
      using_kfac=optim_config.use_kfac,
      strategy=strategy,
      scf_approx=scf_approx,
      global_step=global_step,
      determinism_mode=optim_config.deterministic,
      cached_data_op=update_cached_data,
      write_graph=os.path.abspath(graph_path) if graph_path else None,
      burn_in=mcmc_config.burn_in,
      mcmc_steps=mcmc_config.steps,
  )
