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
"""Top-level QMC training."""

import functools
import os
import shutil
from typing import Iterable, Optional, Text

from absl import flags
from absl import logging
from ferminet import networks
from ferminet.utils import writers
import kfac
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from tensorflow.core.protobuf import rewriter_config_pb2

FLAGS = flags.FLAGS


class QMC:
  """Quantum Monte Carlo on a Fermi Net."""

  def __init__(
      self,
      hamiltonian,
      network,
      data_gen,
      hf_data_gen,
      clip_el: Optional[float] = None,
      check_loss: bool = False,
  ):
    """Constructs a QMC object for training a Fermi Net.

    Args:
      hamiltonian: Hamiltonian operators. See hamiltonian.py
      network: network to represent the wavefunction, psi.
      data_gen: mcmc.MCMC object that samples configurations from
        psi^2, where psi is the output of the network.
      hf_data_gen: mcmc.MCMC object that samples configurations from
        phi^2, where phi is a Hartree-Fock state of the Hamiltonian.
      clip_el: If not None, sets the scale at which to clip the local energy.
        Note the local energy is only clipped in the grad_loss passed back to
        the gradient, not in the local_energy op itself. The width of the window
        in which the local energy is not clipped is a multiple of the total
        variation from the median, which is the equivalent of standard deviation
        for the l1 norm instead of l2 norm. This makes the clipping robuts to
        outliers, which is the entire point.
      check_loss: If True, check the loss for NaNs and raise an error if found.
    """
    self.hamiltonian = hamiltonian
    self.data_gen = data_gen
    self.hf_data_gen = hf_data_gen
    self._clip_el = clip_el
    self._check_loss = check_loss
    self.network = network

  def _qmc_step_fn(self, optimizer_fn, using_kfac, global_step):
    """Training step for network given the MCMC state.

    Args:
      optimizer_fn: A function which takes as argument a LayerCollection object
        (None) if using_kfac is True (False) and returns the optimizer.
      using_kfac: True if optimizer_fn creates a instance of kfac.KfacOptimizer
        and False otherwise.
      global_step: tensorflow op for global step index.

    Returns:
      loss: per-GPU loss tensor with control dependencies for updating network.
      local_energy: local energy for each walker
      features: network output for each walker.

    Raises:
      RuntimeError: If using_kfac is True and optimizer_fn does not create a
      kfac.KfacOptimizer instance or the converse.
    """

    # Note layer_collection cannot be modified after the KFac optimizer has been
    # constructed.
    if using_kfac:
      layer_collection = kfac.LayerCollection()
    else:
      layer_collection = None

    walkers = self.data_gen.walkers_per_gpu
    features, features_sign = self.network(walkers, layer_collection)
    optimizer = optimizer_fn(layer_collection)

    if bool(using_kfac) != isinstance(optimizer, kfac.KfacOptimizer):
      raise RuntimeError('Not using KFac but using_kfac is True.')

    if layer_collection:
      layer_collection.register_squared_error_loss(features, reuse=False)

    with tf.name_scope('local_energy'):
      kinetic_fn, potential_fn = self.hamiltonian
      kinetic = kinetic_fn(features, walkers)
      potential = potential_fn(walkers)
      local_energy = kinetic + potential
      loss = tf.reduce_mean(local_energy)
      replica_context = tf.distribute.get_replica_context()
      mean_op = tf.distribute.ReduceOp.MEAN
      mean_loss = replica_context.merge_call(
          lambda strategy, val: strategy.reduce(mean_op, val), args=(loss,))
      grad_loss = local_energy - mean_loss

      if self._clip_el is not None:
        # clip_el should be much larger than 1, to avoid bias
        median = tfp.stats.percentile(grad_loss, 50.0)
        diff = tf.reduce_mean(tf.abs(grad_loss - median))
        grad_loss_clipped = tf.clip_by_value(grad_loss,
                                             median - self._clip_el * diff,
                                             median + self._clip_el * diff)
      else:
        grad_loss_clipped = grad_loss

    with tf.name_scope('step'):
      # Create functions which take no arguments and return the ops for applying
      # an optimisation step.
      if not optimizer:
        optimize_step = tf.no_op
      else:
        optimize_step = functools.partial(
            optimizer.minimize,
            features,
            global_step=global_step,
            var_list=self.network.trainable_variables,
            grad_loss=grad_loss_clipped)

      if self._check_loss:
        # Apply optimisation step only if all local energies are well-defined.
        step = tf.cond(
            tf.reduce_any(tf.math.is_nan(mean_loss)), tf.no_op, optimize_step)
      else:
        # Faster, but less safe: always apply optimisation step. If the
        # gradients are not well-defined (e.g. loss contains a NaN), then the
        # network will also be set to NaN.
        step = optimize_step()

      # A strategy step function must return tensors, not ops so apply a
      # control dependency to a dummy op to ensure they're executed.
      with tf.control_dependencies([step]):
        loss = tf.identity(loss)

    return {
        'loss': loss,
        'local_energies': local_energy,
        'features': features,
        'features_sign': features_sign
    }

  def train(self,
            optim_fn,
            iterations: int,
            logging_config,
            scf_approx,
            strategy,
            using_kfac: bool,
            prefix: Text = 'fermi_net',
            global_step=None,
            profile_iterations: Optional[Iterable[int]] = None,
            determinism_mode: bool = False,
            cached_data_op=None,
            write_graph: Optional[Text] = None,
            burn_in: int = 0,
            mcmc_steps: int = 10):
    """Training loop for VMC with hooks for logging and plotting.

    Args:
      optim_fn: function with zero arguments (thunk) that returns a class
        implementing optimizer (specifically apply_gradients method).
      iterations: number of iterations to perform for training.
      logging_config: LoggingConfig object defining what and how to log data.
      scf_approx: For hf pretraining, object of class Scf, otherwise None.
      strategy: DistributionStrategy instance to use for parallelisation. Must
        match the strategy used to build the model.
      using_kfac: True if optim_fn creates a kfac.KfacOptimizer object, False
        otherwise.
      prefix: prefix to use for dataframes in bigtable. Appended with _sparse
        and _detailed for configuration and training dataframes respectively.
      global_step: tensorflow op for global step number. If None set to 0 and
        incremented each iteration. Can be used to provide a learning rate
        schedule.
      profile_iterations: iterable of iteration numbers at which the training
        session is to be profiled. Note: data generation step (e.g. MCMC) is not
          included in the profile and must be added separately if required.
      determinism_mode: activates deterministic computation when in CPU mode.
      cached_data_op: Op which updates a variable to cache the current training
        batch. Only necessary if using KFAC with adaptive damping.
      write_graph: if given, write graph to given file. Must be a full path.
      burn_in: Number of burn in steps after pretraining. If zero do not burn in
        or reinitialize walkers.
      mcmc_steps: Number of MCMC steps to perform between network updates.

    Raises:
      RuntimeError: if a NaN value for the loss or another accumulated statistic
      is detected.
    """
    if global_step is None:
      with strategy.scope():
        global_step = tf.train.get_or_create_global_step()
    if profile_iterations is None:
      profile_iterations = []

    if hasattr(self.network, 'pretrain_iterations'):
      stats_len = max(iterations, self.network.pretrain_iterations)
    else:
      stats_len = iterations

    with strategy.scope():
      mcmc_step = self.data_gen.step()
      mcmc_hf_step = self.hf_data_gen.step()
      mcmc_reset_walkers = self.data_gen.reinitialize_walkers()
      mcmc_update_psi = self.data_gen.update_psi()

    # Pretraining
    concat_data = tf.concat([self.hf_data_gen.walkers, self.data_gen.walkers],
                            axis=1)
    pretrain_ops = networks.pretrain_hartree_fock(self.network, concat_data,
                                                  strategy, scf_approx)

    qmc_step_fn = functools.partial(self._qmc_step_fn, optim_fn, using_kfac,
                                    global_step)
    with strategy.scope():
      qmc_out_per_replica = strategy.experimental_run(qmc_step_fn)
    qmc_out_per_replica = {
        key: tf.stack(strategy.experimental_local_results(value))
        for key, value in qmc_out_per_replica.items()
    }
    loss = tf.reduce_mean(qmc_out_per_replica['loss'])

    stats_ops = self.data_gen.stats_ops(log_walkers=logging_config.walkers)
    stats_ops['loss_per_replica'] = qmc_out_per_replica['loss']
    stats_loss_ = np.zeros(shape=qmc_out_per_replica['loss'].shape.as_list())

    if logging_config.wavefunction:
      stats_ops['log_wavefunction'] = qmc_out_per_replica['features']
      stats_ops['wavefunction_sign'] = qmc_out_per_replica['feature_sign']
    if logging_config.local_energy:
      stats_ops['local_energy'] = qmc_out_per_replica['local_energy']

    train_schema = ['energy', 'pmove']
    if logging_config.replicas > 1:
      train_schema += [
          'energy_replica:{}'.format(i) for i in range(logging_config.replicas)
      ]

    # As these data are much larger, we create a separated H5 file to avoid
    # polluting the CSV.
    h5_data = ('walkers', 'log_wavefunction', 'wavefunction_sign',
               'local_energy')
    h5_schema = {
        key: stats_ops[key].shape.as_list()
        for key in h5_data
        if key in stats_ops
    }

    if write_graph:
      tf.train.write_graph(tf.get_default_graph(), os.path.dirname(write_graph),
                           os.path.basename(write_graph))

    if determinism_mode:
      config_proto = tf.ConfigProto(
          inter_op_parallelism_threads=1,
          device_count={'GPU': 0},
          allow_soft_placement=True)
    else:
      config_proto = tf.ConfigProto(allow_soft_placement=True)
    # See https://github.com/tensorflow/tensorflow/issues/23780.
    config_proto.graph_options.rewrite_options.memory_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF)

    saver = tf.train.Saver(max_to_keep=10, save_relative_paths=True)
    scaffold = tf.train.Scaffold(saver=saver)
    checkpoint_path = os.path.join(logging_config.result_path, 'checkpoints')
    if logging_config.restore_path:
      logging.info('Copying %s to %s and attempting to restore',
                   logging_config.restore_path, checkpoint_path)
      shutil.copytree(logging_config.restore_path, checkpoint_path)

    with tf.train.MonitoredTrainingSession(
        scaffold=scaffold,
        checkpoint_dir=checkpoint_path,
        save_checkpoint_secs=logging_config.save_frequency * 60,
        save_checkpoint_steps=None,
        save_summaries_steps=None,
        log_step_count_steps=None,
        config=config_proto,
    ) as sess:
      logging.info('Initialized variables')

      if self.network.pretrain_iterations > 0:
        logging.info('Pretraining network parameters')
        with writers.Writer(
            name='pretrain_stats',
            schema=['loss'],
            directory=logging_config.result_path) as pretrain_writer:
          for t in range(self.network.pretrain_iterations):
            for _ in range(mcmc_steps):
              sess.run(mcmc_step)
            for _ in range(mcmc_steps):
              sess.run(mcmc_hf_step)

            pretrain_loss = sess.run(pretrain_ops)
            pretrain_writer.write(t, loss=pretrain_loss)
            pass
        logging.info('Pretrained network parameters.')

      if burn_in > 0:
        logging.info('Resetting MCMC state')
        sess.run(mcmc_reset_walkers)
        for _ in range(burn_in):
          for _ in range(mcmc_steps):
            sess.run(mcmc_step)
        logging.info('MCMC burn in complete')

      with writers.Writer(name='train_stats', schema=train_schema, directory=logging_config.result_path) as training_writer, \
          writers.H5Writer(name='data.h5', schema=h5_schema, directory=logging_config.result_path) as h5_writer:
        for t in range(iterations):
          if cached_data_op is not None:
            sess.run(cached_data_op)
          sess.run(mcmc_update_psi)
          for _ in range(mcmc_steps):
            sess.run(mcmc_step)
          loss_, stats_ops_ = sess.run([loss, stats_ops])
          stats_values = [loss_] + list(stats_ops_.values())
          # Update the contents (but don't reassign) stats_loss_, so the
          # conditional checkpoint check uses the current per-replica loss via
          # an earlier local-scope capture on stats_loss_.
          stats_loss_[:] = stats_ops_['loss_per_replica']
          if any(np.isnan(stat).any() for stat in stats_values):
            logging.info('NaN value detected. Loss: %s. Stats: %s.', loss_,
                         stats_ops_)
            if self._check_loss:
              # Re-run the mcmc_reset (setting psi) as the checkpoint hook is
              # evaluated at the end of a session run call.
              sess.run(mcmc_update_psi)
            raise RuntimeError(
                'NaN value detected. Loss: {}. Stats: {}.'.format(
                    loss_, stats_ops_))

          if t % logging_config.stats_frequency == 0:
            writer_kwargs = {
                'energy': loss_,
                'pmove': stats_ops_['pmove'],
            }
            if logging_config.replicas > 1:
              for i in range(logging_config.replicas):
                lpr = stats_ops_['loss_per_replica'][i]
                writer_kwargs['energy_replica:%d' % i] = lpr
            training_writer.write(t, **writer_kwargs)
            h5_writer.write(t, stats_ops_)
