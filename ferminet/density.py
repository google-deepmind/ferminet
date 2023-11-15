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

"""Tools for density matrix calculation."""

import functools
from typing import Tuple

from ferminet import constants
from ferminet import mcmc
from ferminet import networks
from ferminet.utils import scf
from jax import numpy as jnp


def _eval_mos(pos: jnp.ndarray, scf_approx: scf.Scf,
              nspins: Tuple[int, int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Evaluates molecular orbitals.

  Args:
    pos: Electron positions of shape (M, 3), where M can be anything.
    scf_approx: SCF object with information about the Hartree-Fock calculation.
    nspins: Number of spin-up and spin-down electrons.

  Returns:
    all_mos: All MOs evaluated at positions `pos`.
    occ_mos: Just the occupied MOs evaluated at those positions.
  """

  if scf_approx.restricted:
    all_mos = jnp.asarray(scf_approx.eval_mos(pos)[0])
    occ_mos = jnp.asarray(all_mos[:, :nspins[0]])
  else:
    all_mos = list(scf_approx.eval_mos(pos))
    occ_mos = jnp.concatenate(
        [mo[:, :nspin] for mo, nspin in zip(all_mos, nspins)], axis=-1)
    all_mos = jnp.concatenate(all_mos, axis=-1)

  return all_mos, occ_mos


def calc_hf_prob(pos: jnp.ndarray, scf_approx: scf.Scf,
                 nspins: Tuple[int, int]) -> jnp.ndarray:
  """Calculates the probability of the current configuration based on Hartree-Fock.
  """

  # evaluate occupied phi's
  _, occ_mos = _eval_mos(
      pos=pos.reshape(-1, 3), scf_approx=scf_approx, nspins=nspins)

  # The probability density of finding a single electron at position r, given a
  # HF wavefunction, is the mean of |phi_i|^2 for each occupied orbital phi_i.
  # See Eq. (2.19) here for proof (after dividing by the number of electrons
  # to convert the electron density to the one-electron probability):
  # https://www.home.uni-osnabrueck.de/apostnik/Lectures/DFT-2.pdf.
  # Intuitively this result says that an electron has the same probability of
  # being in any of the orbitals (electrons are indistinguishable), and the
  # probability density associated with an orbital is |phi_i|^2. So the overall
  # probability is mean(|phi_i|^2).

  # For a restricted calculation we have N / 2 occupied MOs for N electrons,
  # and for an unrestricted calculation we have N (see `_eval_mos`
  # above). So occ_mos has size (batch * N, nocc) where nocc = the number of
  # occupied orbitals = N / 2 for restricted, and N for unrestricted. Then
  # taking jnp.mean(occ_mos ** 2, axis=-1) takes the average over the occupied
  # orbitals, evaluated at the (batch * N) electron positions.

  prob = jnp.mean(occ_mos**2, axis=-1).reshape(pos.shape[:-1])

  return prob


def make_effective_batch_network(
    scf_approx: scf.Scf,
    nspins: Tuple[int, int],
) -> networks.LogFermiNetLike:
  """Makes function to compute 1/2 * log(HF prob) for use in mcmc.make_mcmc_step.

  Args:
    scf_approx: SCF object with information about the Hartree-Fock calculation.
    nspins: Number of spin-up and spin-down electrons.

  Returns:
    eff_batch_network: Function that can be called in the same way as FermiNet
      and return 1/2 * log(probability). This can be used in existing functions
      built for FermiNet, like MCMC steps, but will instead use the probability
      of finding a single electron at a position using the Hartree-Fock
      density.
  """

  def eff_batch_network(params, pos, spins, atoms, charges):
    """Function that can be called like FermiNet, and returns 1/2 * log(HF prob).
    """
    del params, spins, atoms, charges
    prob = calc_hf_prob(pos=pos, scf_approx=scf_approx, nspins=nspins)

    # Since a normal batched network outputs log|psi|, an MCMC step multiplies
    # it by 2 so that it's equal to log|psi|^2 = log(probability). Since we
    # already have the probability here, we need to divide log(prob) by 2 to
    # counteract the multiplication in the MCMC step.

    half_log_prob = 1 / 2 * jnp.log(jnp.abs(prob))
    return half_log_prob

  return eff_batch_network


def make_rprime_mcmc_step(
    steps: int,
    ndim: int,
    blocks: int,
    nspins: Tuple[int, int],
    device_batch_size: int,
    scf_approx: scf.Scf,
) ->...:
  """Makes an MCMC step function for the r' electron positions.

  Args:
    steps: Number of MCMC moves to attempt in a single call to the MCMC step
      function.
    ndim: dimensionality of system.
    blocks: number of blocks to split electron updates into.
    nspins: Number of spin-up and spin-down electrons.
    device_batch_size: Batch size on each device.
    scf_approx: SCF object with information about the Hartree-Fock calculation.

  Returns:
    mcmc_step: A callable function that takes the same arguments as an MCMC
      step created in mcmc.py. The main change is that it also returns the
      probability associated with the last step. This is needed so that we can
      divide by the HF probability when computing the density rho(r, r').
  """
  eff_batch_network = make_effective_batch_network(
      scf_approx=scf_approx, nspins=nspins)
  base_mcmc_step = mcmc.make_mcmc_step(
      eff_batch_network,
      device_batch_size,
      steps=steps,
      blocks=blocks,
      ndim=ndim,
  )

  @functools.partial(constants.pmap)
  def mcmc_step(params, data, mcmc_key, mcmc_width):
    """Regular MCMC step, but with the final probability also returned."""
    # Do the same thing as a regular MCMC step, but also compute the probability
    # at the last step
    data, pmove = base_mcmc_step(params, data, mcmc_key, mcmc_width)
    prob = calc_hf_prob(
        pos=data.positions,
        scf_approx=scf_approx,
        nspins=nspins,
    )
    return data, prob, pmove

  return mcmc_step


def get_rho(
    batch_network: networks.FermiNetLike,
    params: networks.ParamTree,
    dim: int,
    pos: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    nspins: Tuple[int, int],
    batch_atoms: jnp.ndarray,
    rj_pos: jnp.ndarray,
    probs: jnp.ndarray,
    scf_approx: scf.Scf,
) -> jnp.ndarray:
  r"""Gets a sample of the density matrix from r' and (r1, ..., rN) positions.

  The general approach is to construct the density matrix rho(r, r') in a basis
  of MOs {\phi_i}, giving us the matrix rho_ij. We construct samples of rho_ij
  for each MCMC step (after decorrelation) of the positions (r1, ..., rN) from
  the wavefunction, and the position r' from the marginal (one-electron)
  probability from the Hartree-Fock wavefunction. We use the HF wavefunction
  instead of the real wavefunction because the marginal distribution is known
  analytically.

  Explicitly, we have

  rho_ij = N \int dr' dr1 ... drN \phi_i(r1) \phi_j(r') * \psi(r1, r2, ..., rN)
           * \psi(r', r2, ..., rN).

  = N * expectation_{r' ~ p_HF(r'), {r1, ..., rN} ~ |psi(r1, ..., rN)|^2 } (
    \psi(r', ..., rN) \phi_i(r1) \phi_j(r')
    / [\psi(r1, ..., rN) * p_HF(r')] ),

  where
    p_HF(r') = \int dr2, ..., drN |\psi_HF(r', r2, ..., rN)|^2
  is the one-electron probability from Hartree-Fock.

  Args:
    batch_network: vmapped network giving the sign of psi and log|psi|.
    params: Network parameters.
    dim: System dimension.
    pos: (r1, ..., rN) electron positions.
    spins: Spin of each electron.
    charges: Atom charges.
    nspins: Number of spin-up and spin-down electrons.
    batch_atoms: Atom coordinates, replicated along batch dimensions.
    rj_pos: Sampled positions of r' (used in phi_j calculations).
    probs: Probability of sampling the current values of r'.
    scf_approx: Scf object with Hartree-Fock information about the current atom
      configuration.

  Returns:
    rho_mat: Estimate of the one-body density matrix, expressed in a basis of
    Hartree-Fock molecular orbitals.

  Raises:
    ValueError: if system dimension is not 3 or if the number of spin-up
    electrons is not equal to the number of spin-down electrons.
  """

  if dim != 3:
    raise ValueError('Only implemented for 3D systems')

  # Treat spins separately by default
  idx = (0, nspins[0]) if nspins[1] > 0 else (0,)
  rho_mats = []

  denom_signs, denom_logs = batch_network(
      params,
      pos,
      spins,
      batch_atoms,
      charges,
  )

  phi_j, _ = _eval_mos(
      pos=rj_pos.reshape(-1, dim), scf_approx=scf_approx, nspins=nspins)

  use_excited = denom_signs.ndim == 3  # only true for excited states
  if use_excited:
    nstates = denom_signs.shape[-1]
    # Reshape pos to be (batch, states, num_el * dim)
    pos = pos.reshape(pos.shape[0], nstates, -1)
    rj_pos = rj_pos.reshape(-1, nstates, dim)
    probs = probs.reshape(-1, nstates)
    phi_j = phi_j.reshape(-1, nstates, phi_j.shape[-1])

  norb = phi_j.shape[-1] // len(idx)  # number of orbitals per spin

  for spin, i in enumerate(idx):
    sampled_pos = pos.at[..., dim*i:dim*(i+1)].set(rj_pos)
    numer_signs, numer_logs = batch_network(
        params,
        sampled_pos,
        spins,
        batch_atoms,
        charges,
    )

    r1 = pos[..., dim*i:dim*(i+1)].reshape(-1, dim)
    phi_i, _ = _eval_mos(pos=r1, scf_approx=scf_approx, nspins=nspins)
    if use_excited:
      phi_i = phi_i.reshape(-1, nstates, phi_i.shape[-1])

      # subtract off log probs *before* computing log_max for stability
      numer_logs -= jnp.expand_dims(jnp.log(probs), -1)
      log_max = jnp.maximum(jnp.max(denom_logs, axis=[1, 2], keepdims=True),
                            jnp.max(numer_logs, axis=[1, 2], keepdims=True))
      denom = denom_signs * jnp.exp(denom_logs - log_max)
      numer = numer_signs * jnp.exp(numer_logs - log_max)

      phi_i_ = jnp.transpose(phi_i, (0, 2, 1))
      phi_i_ = phi_i_[:, spin*norb:(spin+1)*norb, None, :, None]

      phi_j_ = jnp.transpose(phi_j, (0, 2, 1))
      phi_j_ = phi_j_[:, None, spin*norb:(spin+1)*norb, :, None]

      numer_ = numer[:, None, None] * phi_i_ * phi_j_

      frac = jnp.linalg.solve(denom[:, None, None], numer_)
      rho_mat = jnp.mean(frac, axis=0) * nspins[spin] * (2 // len(idx))
    else:
      frac = numer_signs * denom_signs * jnp.exp(numer_logs - denom_logs)
      norm_frac = frac / probs
      rho_mat = jnp.mean(
          phi_j[:, None, spin*norb:(spin+1)*norb] *
          phi_i[:, spin*norb:(spin+1)*norb, None] *
          norm_frac[:, None, None],
          axis=0) * nspins[spin] * (2 // len(idx))

    rho_mats.append(rho_mat)

  return jnp.stack(rho_mats)
