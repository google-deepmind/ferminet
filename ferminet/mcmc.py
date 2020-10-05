# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Simple Markov Chain Monte Carlo data generator."""

import functools
import tensorflow.compat.v1 as tf


class MCMC:
  """Simple Markov Chain Monte Carlo implementation for sampling from |f|^2."""

  def __init__(self,
               network,
               batch_size,
               init_mu,
               init_sigma,
               move_sigma,
               dtype=tf.float32):
    """Constructs MCMC object.

    Args:
      network: network to sample from according to the square of the output.
      batch_size: batch size - number of configurations (walkers) to generate.
      init_mu: mean of a truncated normal distribution to draw from to generate
        initial configurations for each coordinate (i.e. of length 3 x number of
        electrons).
        distribution.
      init_sigma: standard deviation of a truncated normal distribution to draw
        from to generate initial configurations.
      move_sigma: standard deviation of random normal distribution from which to
        draw random moves.
      dtype: the data type of the configurations.
    """

    strategy = tf.distribute.get_strategy()
    nreplicas = strategy.num_replicas_in_sync
    self._dtype = dtype.base_dtype
    self.network = network
    self._init_mu = init_mu
    self._init_sigma = init_sigma
    self._move_sigma = move_sigma
    self.walkers = tf.get_variable(
        'walkers',
        initializer=self._rand_init((nreplicas, batch_size)),
        trainable=False,
        use_resource=True,
        dtype=self._dtype)
    self.psi = self.update_psi()
    self.move_acc = tf.constant(0.0)

  def _rand_init(self, batch_dims):
    """Generate a random set of samples from a uniform distribution."""
    return tf.concat(
        [
            tf.random.truncated_normal(  # pylint: disable=g-complex-comprehension
                shape=(*batch_dims, 1),
                mean=mu,
                stddev=self._init_sigma,
                dtype=self._dtype) for mu in self._init_mu
        ],
        axis=-1)

  def reinitialize_walkers(self):
    walker_reset = self.walkers.assign(
        self._rand_init(self.walkers.shape.as_list()[:-1]))
    with tf.control_dependencies([walker_reset]):
      psi_reset = self.update_psi()
    return walker_reset, psi_reset

  def update_psi(self):
    strategy = tf.distribute.get_strategy()
    psi_per_gpu = strategy.experimental_run(
        lambda: self.network(self.walkers_per_gpu)[0])
    self.psi = tf.stack(
        strategy.experimental_local_results(psi_per_gpu), axis=0)
    return self.psi

  @property
  def walkers_per_gpu(self):
    replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
    return self.walkers[replica_id]

  @property
  def psi_per_gpu(self):
    replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
    return self.psi[replica_id]

  def step(self):
    """Returns ops for one Metropolis-Hastings step across all replicas."""
    strategy = tf.distribute.get_strategy()
    walkers, psi, move_acc = strategy.experimental_run(
        functools.partial(self._mh_step, step_size=self._move_sigma))
    update_walkers = self.walkers.assign(
        tf.stack(strategy.experimental_local_results(walkers), axis=0))
    self.psi = tf.stack(strategy.experimental_local_results(psi), axis=0)
    self.move_acc = tf.reduce_mean(
        strategy.experimental_local_results(move_acc))
    return update_walkers, self.psi, self.move_acc

  def _mh_step(self, step_size):
    """Returns ops for one Metropolis-Hastings step in a replica context."""
    walkers, psi = self.walkers_per_gpu, self.psi_per_gpu
    new_walkers = walkers + tf.random.normal(
        shape=walkers.shape, stddev=step_size, dtype=self._dtype)
    new_psi = self.network(new_walkers)[0]
    pmove = tf.squeeze(2 * (new_psi - psi))
    pacc = tf.log(
        tf.random_uniform(shape=walkers.shape.as_list()[:1], dtype=self._dtype))
    decision = tf.less(pacc, pmove)
    with tf.control_dependencies([decision]):
      new_walkers = tf.where(decision, new_walkers, walkers)
      new_psi = tf.where(decision, new_psi, psi)
    move_acc = tf.reduce_mean(tf.cast(decision, tf.float32))
    return new_walkers, new_psi, move_acc

  def stats_ops(self, log_walkers=False):
    """Returns op for evaluating the move acceptance probability.

    Args:
      log_walkers: also include the complete walker configurations (slow, lots
      of data).
    """
    stats = {'pmove': self.move_acc}
    if log_walkers:
      stats['walkers'] = self.walkers
    return stats
