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

from typing import Any, Sequence

import chex
from ferminet import networks
import jax
from jax import lax
import jax.numpy as jnp
from typing_extensions import Protocol


class LocalEnergy(Protocol):

  def __call__(self, params: networks.ParamTree, key: chex.PRNGKey,
               data: jnp.ndarray) -> jnp.ndarray:
    """Returns the local energy of a Hamiltonian at a configuration.

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    """


class MakeLocalEnergy(Protocol):

  def __call__(self,
               f: networks.LogFermiNetLike,
               atoms: jnp.ndarray,
               charges: jnp.ndarray,
               spins: Sequence[int],
               use_scan: bool = False,
               **kwargs: Any) -> LocalEnergy:
    """Builds the LocalEnergy function.

    Args:
      f: Callable which evaluates the log of the magnitude of the wavefunction.
      atoms: atomic positions.
      charges: nuclear charges.
      spins: Number of particles of each spin.
      use_scan: Whether to use a `lax.scan` for computing the laplacian.
      **kwargs: additional kwargs to use for creating the specific Hamiltonian.
    """


def local_kinetic_energy(
    f: networks.LogFermiNetLike,
    use_scan: bool = False) -> networks.LogFermiNetLike:
  r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: Callable which evaluates the log of the magnitude of the wavefunction.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.

  Returns:
    Callable which evaluates the local kinetic energy,
    -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
  """

  def _lapl_over_f(params, data):
    n = data.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.grad(f, argnums=1)
    grad_f_closure = lambda x: grad_f(params, x)

    def hessian_diagonal(i):
      primal, tangent = jax.jvp(grad_f_closure, (data,), (eye[i],))
      return primal[i] ** 2 + tangent[i]

    if use_scan:
      _, diagonal = lax.scan(
          lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n)
      return -0.5 * jnp.sum(diagonal)
    else:
      return -0.5 * lax.fori_loop(
          0, n, lambda i, val: val + hessian_diagonal(i), 0.0)

  return _lapl_over_f


def potential_energy(r_ae: jnp.ndarray, r_ee: jnp.ndarray, atoms: jnp.ndarray,
                     charges: jnp.ndarray) -> jnp.ndarray:
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
  v_ee = jnp.sum(jnp.triu(1 / r_ee[..., 0], k=1))
  v_ae = -jnp.sum(charges / r_ae[..., 0])  # pylint: disable=invalid-unary-operand-type
  r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
  v_aa = jnp.sum(
      jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))
  return v_ee + v_ae + v_aa


def local_energy(f: networks.LogFermiNetLike,
                 atoms: jnp.ndarray,
                 charges: jnp.ndarray,
                 spins: Sequence[int],
                 use_scan: bool = False) -> LocalEnergy:
  """Creates the function to evaluate the local energy.

  Args:
    f: Callable which returns the log of the magnitude of the wavefunction given
      the network parameters and configurations data.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
    spins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  del spins
  ke = local_kinetic_energy(f, use_scan=use_scan)

  def _e_l(params: networks.ParamTree, key: chex.PRNGKey,
           data: jnp.ndarray) -> jnp.ndarray:
    """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    del key  # unused
    _, _, r_ae, r_ee = networks.construct_input_features(data, atoms)
    potential = potential_energy(r_ae, r_ee, atoms, charges)
    kinetic = ke(params, data)
    return potential + kinetic

  return _e_l
