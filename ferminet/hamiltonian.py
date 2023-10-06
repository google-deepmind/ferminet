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

"""Evaluating the Hamiltonian on a wavefunction."""

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import chex
from ferminet import networks
from ferminet.utils import utils
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol


Array = Union[jnp.ndarray, np.ndarray]


class LocalEnergy(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the local energy of a Hamiltonian at a configuration.

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    """


class MakeLocalEnergy(Protocol):

  def __call__(
      self,
      f: networks.FermiNetLike,
      charges: jnp.ndarray,
      nspins: Sequence[int],
      use_scan: bool = False,
      complex_output: bool = False,
      **kwargs: Any
  ) -> LocalEnergy:
    """Builds the LocalEnergy function.

    Args:
      f: Callable which evaluates the sign and log of the magnitude of the
        wavefunction.
      charges: nuclear charges.
      nspins: Number of particles of each spin.
      use_scan: Whether to use a `lax.scan` for computing the laplacian.
      complex_output: If true, the output of f is complex-valued.
      **kwargs: additional kwargs to use for creating the specific Hamiltonian.
    """


KineticEnergy = Callable[
    [networks.ParamTree, networks.FermiNetData], jnp.ndarray
]


def local_kinetic_energy(
    f: networks.FermiNetLike,
    use_scan: bool = False,
    complex_output: bool = False,
) -> KineticEnergy:
  r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: Callable which evaluates the wavefunction as a
      (sign or phase, log magnitude) tuple.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.

  Returns:
    Callable which evaluates the local kinetic energy,
    -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
  """

  phase_f = utils.select_output(f, 0)
  logabs_f = utils.select_output(f, 1)

  def _lapl_over_f(params, data):
    n = data.positions.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.grad(logabs_f, argnums=1)
    def grad_f_closure(x):
      return grad_f(params, x, data.spins, data.atoms, data.charges)

    primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)

    if complex_output:
      grad_phase = jax.grad(phase_f, argnums=1)
      def grad_phase_closure(x):
        return grad_phase(params, x, data.spins, data.atoms, data.charges)
      phase_primal, dgrad_phase = jax.linearize(
          grad_phase_closure, data.positions)
      hessian_diagonal = (
          lambda i: dgrad_f(eye[i])[i] + 1.j * dgrad_phase(eye[i])[i]
      )
    else:
      hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

    if use_scan:
      _, diagonal = lax.scan(
          lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n)
      result = -0.5 * jnp.sum(diagonal)
    else:
      result = -0.5 * lax.fori_loop(
          0, n, lambda i, val: val + hessian_diagonal(i), 0.0)
    result -= 0.5 * jnp.sum(primal ** 2)
    if complex_output:
      result += 0.5 * jnp.sum(phase_primal ** 2)
      result -= 1.j * jnp.sum(primal * phase_primal)
    return result

  return _lapl_over_f


def excited_kinetic_energy_matrix(f: networks.FermiNetLike,
                                  states: int) -> KineticEnergy:
  """Creates a f'n which evaluates the matrix of local kinetic energies.

  Args:
    f: A network which returns a tuple of sign(psi) and log(|psi|) arrays, where
      each array contains one element per excited state.
    states: the number of excited states

  Returns:
    A function which computes the matrices (psi) and (K psi), which are the
      value of the wavefunction and the kinetic energy applied to the
      wavefunction for all combinations of electron sets and excited states.
  """

  def _lapl_all_states(params, pos, spins, atoms, charges):
    """Return K psi/psi for each excited state."""
    n = pos.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.jacrev(utils.select_output(f, 1), argnums=1)
    grad_f_closure = lambda x: grad_f(params, x, spins, atoms, charges)
    primal, dgrad_f = jax.linearize(grad_f_closure, pos)

    result = -0.5 * lax.fori_loop(
        0, n, lambda i, val: val + dgrad_f(eye[i])[:, i], jnp.zeros(states))

    return result - 0.5 * jnp.sum(primal ** 2, axis=-1)

  def _lapl_over_f(params, data):
    """Return the kinetic energy (divided by psi) summed over excited states."""
    pos_ = jnp.reshape(data.positions, [states, -1])
    spins_ = jnp.reshape(data.spins, [states, -1])
    vmap_f = jax.vmap(f, (None, 0, 0, None, None))
    sign_mat, log_mat = vmap_f(params, pos_, spins_, data.atoms, data.charges)
    vmap_lapl = jax.vmap(_lapl_all_states, (None, 0, 0, None, None))
    lapl = vmap_lapl(params, pos_, spins_, data.atoms,
                     data.charges)  # K psi_i(r_j) / psi_i(r_j)

    # subtract off largest value to avoid under/overflow
    psi_mat = sign_mat * jnp.exp(log_mat - jnp.max(log_mat))  # psi_i(r_j)
    kpsi_mat = lapl * psi_mat  # K psi_i(r_j)
    return psi_mat, kpsi_mat

  return _lapl_over_f


def potential_electron_electron(r_ee: Array) -> jnp.ndarray:
  """Returns the electron-electron potential.

  Args:
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
  """
  r_ee = r_ee[jnp.triu_indices_from(r_ee[..., 0], 1)]
  return (1.0 / r_ee).sum()


def potential_electron_nuclear(charges: Array, r_ae: Array) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
  """
  return -jnp.sum(charges / r_ae[..., 0])


def potential_nuclear_nuclear(charges: Array, atoms: Array) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    atoms: Shape (natoms, ndim). Positions of the atoms.
  """
  r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
  return jnp.sum(
      jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))


def potential_energy(r_ae: Array, r_ee: Array, atoms: Array,
                     charges: Array) -> jnp.ndarray:
  """Returns the potential energy for this electron configuration.

  Args:
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
  """
  return (potential_electron_electron(r_ee) +
          potential_electron_nuclear(charges, r_ae) +
          potential_nuclear_nuclear(charges, atoms))


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    states: int = 0,
) -> LocalEnergy:
  """Creates the function to evaluate the local energy.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    states: Number of excited states to compute. If 0, compute ground state with
      default machinery. If 1, compute ground state with excited state machinery

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  if complex_output and states > 1:
    raise NotImplementedError(
        'Excited states not implemented with complex output')
  del nspins
  if states:
    ke = excited_kinetic_energy_matrix(f, states)
  else:
    ke = local_kinetic_energy(f,
                              use_scan=use_scan,
                              complex_output=complex_output)

  def _e_l(
      params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    del key  # unused
    if states:
      # Compute features
      vmap_features = jax.vmap(networks.construct_input_features, (0, None))
      positions = jnp.reshape(data.positions, [states, -1])
      _, _, r_ae, r_ee = vmap_features(positions, data.atoms)

      # Compute potential energy
      vmap_pot = jax.vmap(potential_energy, (0, 0, None, None))
      pot_spectrum = vmap_pot(r_ae, r_ee, data.atoms, charges)[:, None]

      # Compute kinetic energy and matrix of states
      psi_mat, kin_mat = ke(params, data)

      # Combine terms
      hpsi_mat = kin_mat + psi_mat * pot_spectrum
      energy_mat = jnp.linalg.solve(psi_mat, hpsi_mat)
      total_energy = jnp.trace(energy_mat)
    else:
      _, _, r_ae, r_ee = networks.construct_input_features(
          data.positions, data.atoms
      )
      potential = potential_energy(r_ae, r_ee, data.atoms, charges)
      kinetic = ke(params, data)
      total_energy = potential + kinetic
      energy_mat = None  # Not necessary for ground state
    return total_energy, energy_mat

  return _e_l
