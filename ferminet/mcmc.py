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

"""Metropolis-Hastings Monte Carlo.

NOTE: these functions operate on batches of MCMC configurations and should not
be vmapped.
"""

from ferminet import constants
import jax
from jax import lax
from jax import numpy as jnp


def _harmonic_mean(x, atoms):
  """Calculates the harmonic mean of each electron distance to the nuclei.

  Args:
    x: electron positions. Shape (batch, nelectrons, 1, ndim). Note the third
      dimension is already expanded, which allows for avoiding additional
      reshapes in the MH algorithm.
    atoms: atom positions. Shape (natoms, ndim)

  Returns:
    Array of shape (batch, nelectrons, 1, 1), where the (i, j, 0, 0) element is
    the harmonic mean of the distance of the j-th electron of the i-th MCMC
    configuration to all atoms.
  """
  ae = x - atoms[None, ...]
  r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
  return 1.0 / jnp.mean(1.0 / r_ae, axis=-2, keepdims=True)


def _log_prob_gaussian(x, mu, sigma):
  """Calculates the log probability of Gaussian with diagonal covariance.

  Args:
    x: Positions. Shape (batch, nelectron, 1, ndim) - as used in mh_update.
    mu: means of Gaussian distribution. Same shape as or broadcastable to x.
    sigma: standard deviation of the distribution. Same shape as or
      broadcastable to x.

  Returns:
    Log probability of Gaussian distribution with shape as required for
    mh_update - (batch, nelectron, 1, 1).
  """
  numer = jnp.sum(-0.5 * ((x - mu)**2) / (sigma**2), axis=[1, 2, 3])
  denom = x.shape[-1] * jnp.sum(jnp.log(sigma), axis=[1, 2, 3])
  return numer - denom


def mh_update(params,
              f,
              x1,
              key,
              lp_1,
              num_accepts,
              stddev=0.02,
              atoms=None,
              i=0,
              ndim=3):
  """Performs one Metropolis-Hastings step using an all-electron move.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    x1: Initial MCMC configurations. Shape (batch, nelectrons*ndim).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    atoms: If not None, atom positions. Shape (natoms, 3). If present, then the
      Metropolis-Hastings move proposals are drawn from a Gaussian distribution,
      N(0, (h_i stddev)^2), where h_i is the harmonic mean of distances between
      the i-th electron and the atoms, otherwise the move proposal drawn from
      N(0, stddev^2).
    i: Ignored.

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  del i  # electron index ignored for all-electron moves
  key, subkey = jax.random.split(key)
  if atoms is None:  # symmetric proposal, same stddev everywhere
    x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
    lp_2 = 2. * f(params, x2)  # log prob of proposal
    ratio = lp_2 - lp_1
  else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
    n = x1.shape[0]
    x1 = jnp.reshape(x1, [n, -1, 1, ndim])
    hmean1 = _harmonic_mean(x1, atoms)  # harmonic mean of distances to nuclei

    x2 = x1 + stddev * hmean1 * jax.random.normal(subkey, shape=x1.shape)
    lp_2 = 2. * f(params, x2)  # log prob of proposal
    hmean2 = _harmonic_mean(x2, atoms)  # needed for probability of reverse jump

    lq_1 = _log_prob_gaussian(x1, x2, stddev * hmean1)  # forward probability
    lq_2 = _log_prob_gaussian(x2, x1, stddev * hmean2)  # reverse probability
    ratio = lp_2 + lq_2 - lp_1 - lq_1

    x1 = jnp.reshape(x1, [n, -1])
    x2 = jnp.reshape(x2, [n, -1])
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=lp_1.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)

  return x_new, key, lp_new, num_accepts


def mh_one_electron_update(params,
                           f,
                           x1,
                           key,
                           lp_1,
                           num_accepts,
                           stddev=0.02,
                           atoms=None,
                           i=0,
                           ndim=3):
  """Performs one Metropolis-Hastings step for a single electron.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    x1: Initial MCMC configurations. Shape (batch, nelectrons*ndim).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    atoms: Ignored. Asymmetric move proposals are not implemented for
      single-electron moves.
    i: index of electron to move.

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.

  Raises:
    NotImplementedError: if atoms is supplied.
  """
  key, subkey = jax.random.split(key)
  n = x1.shape[0]
  x1 = jnp.reshape(x1, [n, -1, 1, ndim])
  nelec = x1.shape[1]
  ii = i % nelec
  if atoms is None:  # symmetric proposal, same stddev everywhere
    x2 = x1.at[:, ii].add(stddev *
                          jax.random.normal(subkey, shape=x1[:, ii].shape))
    lp_2 = 2. * f(params, x2)  # log prob of proposal
    ratio = lp_2 - lp_1
  else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
    raise NotImplementedError('Still need to work out reverse probabilities '
                              'for asymmetric moves.')

  x1 = jnp.reshape(x1, [n, -1])
  x2 = jnp.reshape(x2, [n, -1])
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=lp_1.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)

  return x_new, key, lp_new, num_accepts


def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   atoms=None,
                   one_electron_moves=False,
                   ndim=3):
  """Creates the MCMC step function.

  Args:
    batch_network: function, signature (params, x), which evaluates the log of
      the wavefunction (square root of the log probability distribution) at x
      given params. Inputs and outputs are batched.
    batch_per_device: Batch size per device.
    steps: Number of MCMC moves to attempt in a single call to the MCMC step
      function.
    atoms: atom positions. If given, an asymmetric move proposal is used based
      on the harmonic mean of electron-atom distances for each electron.
      Otherwise the (conventional) normal distribution is used.
    one_electron_moves: If true, attempt to move one electron at a time.
      Otherwise, attempt one all-electron move per MCMC step.

  Returns:
    Callable which performs the set of MCMC steps.
  """
  inner_fun = mh_one_electron_update if one_electron_moves else mh_update

  @jax.jit
  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.
      key: RNG state.
      width: standard deviation to use in the move proposal.

    Returns:
      (data, pmove), where data is the updated MCMC configurations, key the
      updated RNG state and pmove the average probability a move was accepted.
    """

    def step_fn(i, x):
      return inner_fun(
          params, batch_network, *x, stddev=width, atoms=atoms, i=i, ndim=ndim)

    nelec = data.shape[-1] // ndim
    nsteps = nelec * steps if one_electron_moves else steps
    logprob = 2. * batch_network(params, data)
    data, key, _, num_accepts = lax.fori_loop(0, nsteps, step_fn,
                                              (data, key, logprob, 0.))
    pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
    pmove = constants.pmean(pmove)
    return data, pmove

  return mcmc_step
