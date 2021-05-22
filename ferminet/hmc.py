# Copyright 2021 DeepMind Technologies Limited.
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

"""Hamiltonian Monte Carlo"""

from jax import grad
import jax
from jax import lax
import jax.numpy as jnp
from ferminet import constants

from kfac_ferminet_alpha import utils as kfac_utils

def leapfrog(q0, p0, dv_dq, path_len, step_size):
  """Updates the q and p variables by performing path_len/step_size - 1 steps.

  Args:
    q0: The latest sample in the chain.
    p0: The sampled momentum.
    dv_dq: The gradient of the distribution.
    path_len: The intended length of the leapfrog path.
    step_size: The size of each leapfrog step.

  Returns:
    q1: The new sample after taking leapfrog steps.
    p1: The new momentum after taking leapfrog steps.
  """
  def leapfrog_step(params):
    (i, q1, p1) = params
    q1 += step_size * p1
    p1 -= step_size *dv_dq(q1)
    return (i+1, q1, p1)

  q1, p1 = q0, p0

  p1 -= step_size * dv_dq(q1) / 2  # starting half step
  _, q1, p1 = jax.lax.while_loop(
    lambda x: x[0] < (int(path_len / step_size) - 1),
    leapfrog_step, (0, q1, p1))

  q1 += step_size * p1
  p1 -= step_size *dv_dq(q1) / 2  # final half step

  return q1, -p1

def hmc_samples(distribution):
  """Collects one sample from the distribution.

  Args:
    distribution: function, signature (params, x), which evaluates the log of
      the wavefunction (square root of the log probability distribution) at x
      given params. Inputs and outputs are not batched.

  Returns:
    Callable which collects one sample from the distribution.
  """

  def _hmc_samples(num_samples, start_pos, path_len, step_size, key):
    """Inner function to sample from the distribution using HMC.

    Args:
      num_samples: Number of samples to collect in the chain.
      start_pos: First sample in the chain.
      path_len: The length of the leapfrog path when collecting the next sample.
      step_size: The size of the leapfrog steps.
      key: The RNG state.

    Returns:
      (data, pmove), where data is the updated HMC configurations and pmove
      the average probability a move was accepted.
    """
    def inner_fun(_, params):
      (last_sample, num_accepts, key) = params
      dv_dq = grad(distribution)

      key, subkey1, subkey2 = jax.random.split(key, num=3)
      p0 = jax.random.normal(subkey1, shape=start_pos.shape)

      # Integrate using leapfrog to get pos and momentum
      q1, p1 = leapfrog(
        last_sample,
        p0,
        dv_dq,
        path_len,
        step_size
      )
      # accept based on metropolis acceptance criterion
      dist_p_start = (distribution(last_sample)
                      - jnp.sum(jax.scipy.stats.norm.logpdf(p0)))
      dist_p_new = distribution(q1) - jnp.sum(jax.scipy.stats.norm.logpdf(p1))
      move = lax.cond(
        jnp.log(jax.random.uniform(subkey2)) < dist_p_new - dist_p_start,
        lambda _: True,
        lambda _: False,
        operand=None)

      num_accepts += move

      new_sample = lax.cond(
        move,
        lambda _: q1,
        lambda _: jnp.array(last_sample, copy=True),
        operand=None)

      return new_sample, num_accepts, key

    # keep track of last sample
    last_sample = start_pos

    num_accepts = 0
    # Need num_samples draws from momentum
    last_sample, num_accepts, key = jax.lax.fori_loop(
      0, num_samples, inner_fun, (last_sample, num_accepts, key))

    return (last_sample, num_accepts)

  return jax.vmap(_hmc_samples, in_axes=(None, 0, None, None, 0))

def make_hmc_step(network,
                  batch_per_device,
                  steps=5,
                  step_size=.1,
                  path_len=1):
  """Creates the HMC step function.

  Args:
    network: function, signature (params, x), which evaluates the log of
      the wavefunction (square root of the log probability distribution) at x
      given params. Inputs and outputs are not batched.
    batch_per_device: Batch size per device.
    steps: Number of HMC steps to attempt in a single call to the HMC step
      function.
    step_size: The size of the leapfrog steps.
    path_len: The length of the leapfrog path when collecting the next sample.

  Returns:
    Callable which performs the set of MCMC steps.
  """
  def hmc_step(params, data, key, width):
    """Performs a set of HMC steps.

    Args:
      params: parameters to pass to the network.
      data: (batched) HMC configurations to pass to the network.
      key: RNG state.
      width: unused param, to ensure consistency with the MCMC code

    Returns:
      (data, pmove), where data is the updated HMC configurations and
      pmove the average probability a move was accepted.
    """
    del width

    def neg_logprob(x):
      return -2. * network(params, x)

    key_per_batch = jax.random.split(key, num=batch_per_device)
    samples, num_accepts = hmc_samples(neg_logprob)(steps, data, path_len,
                                                    step_size, key_per_batch)
    pmove = jnp.sum(num_accepts) / (steps * batch_per_device)
    pmove = kfac_utils.pmean_if_pmap(pmove, axis_name=constants.PMAP_AXIS_NAME)
    return samples, pmove

  return hmc_step
