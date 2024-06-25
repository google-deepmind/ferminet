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

import chex
from ferminet import constants
from ferminet import networks
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np


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


def mh_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts):
  """Given state, proposal, and probabilities, execute MH accept/reject step."""
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)
  return x_new, key, lp_new, num_accepts


def mh_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    atoms=None,
    ndim=3,
    blocks=1,
    i=0,
):
  """Performs one Metropolis-Hastings step using an all-electron move.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    atoms: If not None, atom positions. Shape (natoms, 3). If present, then the
      Metropolis-Hastings move proposals are drawn from a Gaussian distribution,
      N(0, (h_i stddev)^2), where h_i is the harmonic mean of distances between
      the i-th electron and the atoms, otherwise the move proposal drawn from
      N(0, stddev^2).
    ndim: dimensionality of system.
    blocks: Ignored.
    i: Ignored.

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  del i, blocks  # electron index ignored for all-electron moves
  key, subkey = jax.random.split(key)
  x1 = data.positions
  if atoms is None:  # symmetric proposal, same stddev everywhere
    x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
    lp_2 = 2.0 * f(
        params, x2, data.spins, data.atoms, data.charges
    )  # log prob of proposal
    ratio = lp_2 - lp_1
  else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
    n = x1.shape[0]
    x1 = jnp.reshape(x1, [n, -1, 1, ndim])
    hmean1 = _harmonic_mean(x1, atoms)  # harmonic mean of distances to nuclei

    x2 = x1 + stddev * hmean1 * jax.random.normal(subkey, shape=x1.shape)
    lp_2 = 2.0 * f(
        params, x2, data.spins, data.atoms, data.charges
    )  # log prob of proposal
    hmean2 = _harmonic_mean(x2, atoms)  # needed for probability of reverse jump

    lq_1 = _log_prob_gaussian(x1, x2, stddev * hmean1)  # forward probability
    lq_2 = _log_prob_gaussian(x2, x1, stddev * hmean2)  # reverse probability
    ratio = lp_2 + lq_2 - lp_1 - lq_1

    x1 = jnp.reshape(x1, [n, -1])
    x2 = jnp.reshape(x2, [n, -1])
  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
  return new_data, key, lp_new, num_accepts


def mh_block_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    atoms=None,
    ndim=3,
    blocks=1,
    i=0,
):
  """Performs one Metropolis-Hastings step for a block of electrons.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with LogFermiNetLike signature which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configuration (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    atoms: Not implemented. Raises an error if not None.
    ndim: dimensionality of system.
    blocks: number of blocks to split electron updates into.
    i: index of block of electrons to move.

  Returns:
    (x, key, lp, num_accepts), where:
      x: MCMC configurations with updated positions.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.

  Raises:
    NotImplementedError: if atoms is supplied.
  """
  key, subkey = jax.random.split(key)
  batch_size = data.positions.shape[0]
  nelec = data.positions.shape[1] // ndim
  pad = (blocks - nelec % blocks) % blocks
  x1 = jnp.reshape(
      jnp.pad(data.positions, ((0, 0), (0, pad * ndim))),
      [batch_size, blocks, -1, ndim],
  )
  ii = i % blocks
  if atoms is None:  # symmetric prop, same stddev everywhere
    x2 = x1.at[:, ii].add(
        stddev * jax.random.normal(subkey, shape=x1[:, ii].shape))
    x2 = jnp.reshape(x2, [batch_size, -1])
    if pad > 0:
      x2 = x2[..., :-pad*ndim]
    # log prob of proposal
    lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
    ratio = lp_2 - lp_1
  else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
    raise NotImplementedError('Still need to work out reverse probabilities '
                              'for asymmetric moves.')

  x1 = jnp.reshape(x1, [batch_size, -1])
  if pad > 0:
    x1 = x1[..., :-pad*ndim]
  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
  return new_data, key, lp_new, num_accepts


def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   atoms=None,
                   ndim=3,
                   blocks=1):
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
    ndim: Dimensionality of the system (usually 3).
    blocks: Number of blocks to split the updates into. If 1, use all-electron
      moves.

  Returns:
    Callable which performs the set of MCMC steps.
  """
  inner_fun = mh_block_update if blocks > 1 else mh_update

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
    pos = data.positions

    def step_fn(i, x):
      return inner_fun(
          params,
          batch_network,
          *x,
          stddev=width,
          atoms=atoms,
          ndim=ndim,
          blocks=blocks,
          i=i)

    nsteps = steps * blocks
    logprob = 2.0 * batch_network(
        params, pos, data.spins, data.atoms, data.charges
    )
    new_data, key, _, num_accepts = lax.fori_loop(
        0, nsteps, step_fn, (data, key, logprob, 0.0)
    )
    pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
    pmove = constants.pmean(pmove)
    return new_data, pmove

  return mcmc_step


def update_mcmc_width(
    t: int,
    width: jnp.ndarray,
    adapt_frequency: int,
    pmove: jnp.ndarray,
    pmoves: np.ndarray,
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
) -> tuple[jnp.ndarray, np.ndarray]:
  """Updates the width in MCMC steps.

  Args:
    t: Current step.
    width: Current MCMC width.
    adapt_frequency: The number of iterations after which the update is applied.
    pmove: Acceptance ratio in the last step.
    pmoves: Acceptance ratio over the last N steps, where N is the number of
      steps between MCMC width updates.
    pmove_max: The upper threshold for the range of allowed pmove values
    pmove_min: The lower threshold for the range of allowed pmove values

  Returns:
    width: Updated MCMC width.
    pmoves: Updated `pmoves`.
  """

  t_since_mcmc_update = t % adapt_frequency
  # update `pmoves`; `pmove` should be the same across devices
  pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
  if t > 0 and t_since_mcmc_update == 0:
    if np.mean(pmoves) > pmove_max:
      width *= 1.1
    elif np.mean(pmoves) < pmove_min:
      width /= 1.1
  return width, pmoves
