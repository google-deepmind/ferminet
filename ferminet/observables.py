# Copyright 2023 DeepMind Technologies Limited.
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

"""Monte Carlo evaluation of quantum observables."""
from typing import Any, Callable, Optional, Tuple

import chex
from ferminet import constants
from ferminet import density
from ferminet import mcmc
from ferminet import networks
from ferminet.utils import scf
import jax
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np


@chex.dataclass
class DensityState:
  """The state of the MCMC chain used to compute the density matrix.

  Attributes:
    t: time step, not distributed over devices
    positions: walker positions, shape (devices, nelectrons, ndim).
    probabilities: walker probabilities, shape (devices, nelectrons)
    move_width: MCMC move width, shape (devices)
    pmove: probability of move being accepted, shape (update_frequency)
    mo_coeff: coefficients of Scf approximation, needed for checkpointing. Not
      distributed over devices.
  """

  # Similarly to networks.FermiNetData, this dataclass needs to be used for
  # vmap/pmap in_axes arguments, and so the types need to be flexible.
  t: Any  # int
  positions: Any  # array
  probabilities: Any  # array
  move_width: Any  # array
  pmove: Any  # array
  mo_coeff: Any  # array


Observable = Callable[[networks.ParamTree,
                       networks.FermiNetData,
                       Optional[DensityState]],
                      jnp.ndarray]

DensityUpdate = Callable[[Any,
                          networks.ParamTree,
                          networks.FermiNetData,
                          DensityState],
                         DensityState]


def make_observable_fns(fns, pmap: bool = True):
  """Transform batchless functions to functions averaged over a batch.

  Args:
    fns: arbitrary structure of functions with signatures fn(params, x) ->
      results, where params is the network parameters, x is a configuration of
      electron positions, and result is a scalar.
    pmap: if true, also apply a pmap transformation and take the mean over the
      pmap.

  Returns:
    corresponding structure of functions which act on a batch of electron
    configurations and return the mean over the batch.
  """

  def transform(fn):
    data_axes = networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0)
    batch_fn = jax.vmap(fn, in_axes=(None, data_axes, 0))
    if pmap:
      def mean_fn(params, data, state):
        batch_vals = batch_fn(params, data, state)
        return constants.pmean(jnp.nanmean(batch_vals, axis=0))

      # only return from first device, since pmean makes them all the same
      return lambda *args: constants.pmap(mean_fn)(*args)[0]
    else:
      return lambda *args: jnp.nanmean(batch_fn(*args), axis=0)

  return jax.tree_util.tree_map(transform, fns)


def make_s2(
    signed_network: networks.FermiNetLike,
    nspins: Tuple[int, ...],
    assign_spin: bool = True,
    states: int = 0,
) -> Observable:
  """Evaluates the S^2 operator of a wavefunction.

  See Wang et al, J. Chem. Phys. 102, 3477 (1995).

  Args:
    signed_network: network callable which takes the network parameters and a
      single (batchless) configuration of electrons and returns the sign and log
      of the network.
    nspins: tuple of spin up and down electrons.
    assign_spin: True if the spin configuration (S_z) is fixed, false if sampled
    states: Number of excited states (0 if doing conventional ground state VMC,
      1 if doing ground state VMC using machinery for excited states).

  Returns:
    callable with same arguments as the network and returns the contribution to
    the Monte Carlo estimate of the S^2 expectation value of the wavefunction at
    the given configuration of electrons.
  """

  def s2_estimator(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      state: None = None,
  ) -> jnp.ndarray:
    """Returns the S^2 contribution from a single electron configuration x."""
    del state  # only included for consistency in the interface

    if sum(nspins) == 1:  # one-electron case is trivial - always singlet
      return jnp.eye(states) * 0.75 if states else jnp.asarray(0.75)

    if assign_spin:
      # Without loss of generality, N_a >= N_b.
      # <S^2> = (N_a - N_b)/2 ((N_a - N_b)/2 +1) + N_b
      #          + 2 \int \Gamma^{abba}(r_1 r_2|r_1 r_2) dr_1 dr_2
      # For S=M_s, M_s = (N_a - N_b)/2, the first term is the exact value of
      # S^2. The spin contamination is hence the incomplete cancellation of
      # the second and third terms.
      # Compute the contribution to S^2 from the diagonal pair densities.
      na, nb = sorted(nspins, reverse=True)
      s2_diagonal = (na - nb) / 2 * ((na - nb) / 2 + 1) + nb
      s2 = s2_diagonal
    else:
      # Following Löwdin 1955, the action of S^2 on psi with generic spins is
      # given by -N(N-4)/4 psi + \sum_{i<j} psi_ij, where psi_ij is the
      # wavef'n with spins of i and j swapped.
      n = sum(nspins)
      s2 = -n * (n - 4) / 4

    if states:
      state_matrix_network = networks.make_state_matrix(signed_network, states)
      sign_psi, log_psi = state_matrix_network(
          params, data.positions, data.spins, data.atoms, data.charges)
      log_psi_max = jnp.max(log_psi)
      psi = sign_psi * jnp.exp(log_psi - log_psi_max)  # avoid underflow
    else:
      sign_psi, log_psi = signed_network(params, data.positions, data.spins,
                                         data.atoms, data.charges)

    if assign_spin:
      if states:
        # Instead of evaluating local operator E_x[\psi^-1(x) O \psi(x)],
        # evaluate the matrices A_ij = O \psi_i(x_j) and B_ij = \psi_i(x_j) and
        # return E[A @ B^-1], analogous to how the matrix of local energies is
        # computed.
        s2 = s2 * psi  # promote s2 from a scalar to a matrix
        xa, xb = jnp.split(
            jnp.reshape(data.positions, (states, sum(nspins), -1)),
            nspins[:1], axis=-2)
        def _inner(ib, val):
          ia, s2 = val
          xx = xa.at[:, ia].set(xb[:, ib]), xb.at[:, ib].set(xa[:, ia])
          xx = jnp.reshape(jnp.concatenate(xx, axis=1), -1)
          sign_psi_swap, log_psi_swap = state_matrix_network(
              params, xx, data.spins, data.atoms, data.charges)
          # Minus sign from reordering electron positions such that alpha
          # electrons come first.
          # out to be numerically unstable.
          s2 -= sign_psi_swap * jnp.exp(log_psi_swap - log_psi_max)
          return ia, s2
      else:
        # Convert to (nalpha, ndim) and (nbeta, ndim) tensors.
        xa, xb = jnp.split(
            jnp.reshape(data.positions, (sum(nspins), -1)), nspins[:1], axis=-2)
        def _inner(ib, val):
          ia, s2 = val
          xx = xa.at[ia].set(xb[ib]), xb.at[ib].set(xa[ia])
          xx = jnp.reshape(jnp.concatenate(xx), -1)
          sign_psi_swap, log_psi_swap = signed_network(params, xx, data.spins,
                                                       data.atoms, data.charges)
          # Minus sign from reordering electron positions such that alpha
          # electrons come first.
          s2 -= sign_psi * sign_psi_swap * jnp.exp(log_psi_swap - log_psi)
          return ia, s2

      def _outer(ia, s2):
        return jax.lax.fori_loop(0, nspins[1], _inner, (ia, s2))[1]

      s2 = jax.lax.fori_loop(0, nspins[0], _outer, s2)

      if states:
        s2 = jnp.linalg.solve(psi, s2)
    else:
      if states:
        raise NotImplementedError('S^2 estimation for excited states is only '
                                  'implemented for spin-assigned wavefunctions')
      def _inner(ib, val):
        ia, s2 = val
        xx = data.spins.at[ia].set(data.spins[ib])
        xx = xx.at[ib].set(data.spins[ia])
        sign_psi_swap, log_psi_swap = signed_network(
            params, data.positions, xx, data.atoms, data.charges)
        # Unlike in the fixed-spin case, the wavefunction has no privileged
        # ordering of electrons due to their spin.
        s2 += sign_psi * sign_psi_swap * jnp.exp(log_psi_swap - log_psi)
        return ia, s2

      def _outer(ia, s2):
        return jax.lax.fori_loop(0, ia, _inner, (ia, s2))[1]

      s2 = jax.lax.fori_loop(0, n, _outer, s2)

    return s2

  return s2_estimator


def make_dipole(
    signed_network: networks.FermiNetLike,
    states: int = 0,
) -> Observable:
  """Evaluates the dipole moment of a wavefunction.

  Args:
    signed_network: network callable which takes the network parameters and a
      single (batchless) configuration of electrons and returns the sign and log
      of the network.
    states: Number of excited states (0 if doing conventional ground state VMC,
      1 if doing ground state VMC using machinery for excited states).

  Returns:
    callable with same arguments as the network and returns the contribution to
    the Monte Carlo estimate of the dipole moment of the wavefunction at the
    given configuration of electrons.
  """

  def dipole_estimator(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      state: None = None,
  ) -> jnp.ndarray:
    """Returns the dipole moment from a single electron configuration x."""
    del state  # only included for consistency in the interface
    if states:
      state_matrix_network = networks.make_state_matrix(signed_network, states)
      sign_psi, log_psi = state_matrix_network(
          params, data.positions, data.spins, data.atoms, data.charges)
      log_psi_max = jnp.max(log_psi)
      psi = sign_psi * jnp.exp(log_psi - log_psi_max)  # avoid underflow
      mean_pos = jnp.sum(jnp.reshape(data.positions, (states, -1, 3)), axis=1)
      moment = jnp.linalg.solve(
          psi[None], jnp.einsum('ij,ik->jik', mean_pos, psi))
    else:
      # the dipole moment is trivial in this case - it's just the expected
      # position of an electron, or the center of mass of the electron density
      moment = jnp.sum(jnp.reshape(data.positions, (-1, 3)), axis=0)

    return moment

  return dipole_estimator


def make_density_matrix(
    signed_network: networks.FermiNetLike,
    pos: jnp.ndarray,
    cfg: ml_collections.ConfigDict,
    ckpt_state: Optional[DensityState] = None) -> Tuple[
        DensityState, DensityUpdate, Observable]:
  """Evaluates the density matrix of a wavefunction.

  Args:
    signed_network: network callable which takes the network parameters and a
      single (batchless) configuration of electrons and returns the sign and log
      of the network.
    pos: Position of the electron walkers, used to initialize a parallel chain
      of walkers needed for Monte Carlo estimates of the one-electron reduced
      density matrix.
    cfg: Config for the experiment.
    ckpt_state: Optional state loaded from checkpoint

  Returns:
    The initial state of the parallel MCMC chain, a function to update the chain
    and a function that computes a single sample of the Monte Carlo estimate at
    a given electron position and parallel chain position.
  """

  nstates = cfg.system.states or 1
  device_batch_size = pos.shape[1]

  scf_approx = scf.Scf(
      molecule=cfg.system.molecule,
      restricted=False,  # compute dm for both spins
      nelectrons=cfg.system.electrons,
      basis=cfg.observables.density_basis)
  scf_approx.run()

  if ckpt_state is not None:
    t = ckpt_state.t
    rprime_pos = ckpt_state.positions
    rprime_prob = ckpt_state.probabilities
    move_width = ckpt_state.move_width
    pmove = ckpt_state.pmove
    scf_approx.mo_coeff = ckpt_state.mo_coeff
  else:
    t = 0
    pos = pos.copy().reshape(-1, 3)
    data_shape = (jax.local_device_count(), device_batch_size * nstates)
    idx = np.random.randint(
        low=0,
        high=pos.shape[0] - 1,
        size=np.prod(data_shape),
    )

    # r' positions
    # In the case of excited states, keep the array flat
    rprime_pos = pos[idx].reshape(*data_shape, -1)
    rprime_prob = jnp.ones(rprime_pos.shape[:-1])
    # MCMC move width for r' Monte Carlo sampling
    move_width = kfac_jax.utils.replicate_all_local_devices(jnp.asarray([0.1]))
    pmove = np.zeros(cfg.mcmc.adapt_frequency)

  density_state = DensityState(t=t,
                               positions=rprime_pos,
                               probabilities=rprime_prob,
                               move_width=move_width,
                               pmove=pmove,
                               mo_coeff=scf_approx.mo_coeff)

  rprime_step = density.make_rprime_mcmc_step(
      steps=cfg.mcmc.steps,
      ndim=cfg.system.ndim,
      blocks=1,
      nspins=cfg.system.electrons,
      device_batch_size=device_batch_size * nstates,
      scf_approx=scf_approx,
  )

  if cfg.system.states:
    signed_net = networks.make_state_matrix(signed_network, cfg.system.states)
  else:
    signed_net = signed_network
  batch_signed_net = jax.vmap(
      signed_net, in_axes=(None, 0, 0, 0, 0), out_axes=0,)

  def density_update(key,
                     params: networks.ParamTree,
                     data: networks.FermiNetData,
                     state: DensityState) -> DensityState:
    for _ in range(1000 if state.t == 0 else 1):  # burn-in on the first step
      key, mcmc_key = kfac_jax.utils.p_split(key)

      rprime_data = networks.FermiNetData(
          positions=state.positions,
          spins=data.spins,
          atoms=data.atoms,
          charges=data.charges,
      )

      rprime_data, rprime_probs, rprime_pmove = rprime_step(
          params=params,
          data=rprime_data,
          mcmc_key=mcmc_key,
          mcmc_width=state.move_width)

      move_width, pmoves = mcmc.update_mcmc_width(state.t,
                                                  state.move_width,
                                                  cfg.mcmc.adapt_frequency,
                                                  rprime_pmove,
                                                  state.pmove)
    return DensityState(t=state.t+1,
                        positions=rprime_data.positions,
                        probabilities=rprime_probs,
                        move_width=move_width,
                        pmove=pmoves,
                        mo_coeff=state.mo_coeff)

  def density_estimator(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      state: DensityState,
  ) -> jnp.ndarray:
    return density.get_rho(
        batch_signed_net,
        params,
        cfg.system.ndim,
        data.positions,
        data.spins,
        data.charges,
        cfg.system.electrons,
        data.atoms,
        state.positions,
        state.probabilities,
        scf_approx)

  return density_state, density_update, density_estimator
