# Copyright 2022 DeepMind Technologies Limited.
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
# limitations under the License

"""2D Ewald summation of Coulomb Hamiltonian in periodic boundary conditions.
"""

import itertools
from typing import Callable

import jax
import jax.numpy as jnp


def make_ewald_potential(
    lattice: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    truncation_limit: int = 5,
    include_heg_background: bool = True
) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
  """Creates a function to evaluate infinite Coulomb sum for periodic lattice.

  Args:
    lattice: Shape (2, 2). Matrix whose columns are the primitive lattice
      vectors.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
    truncation_limit: Integer. Half side length of cube of nearest neighbours
      to primitive cell which are summed over in evaluation of Ewald sum.
      Must be large enough to achieve convergence for the real and reciprocal
      space sums.
    include_heg_background: bool. When True, includes cell-neutralizing
      background term for homogeneous electron gas.

  Returns:
    Callable with signature f(ae, ee), where (ae, ee) are atom-electon and
    electron-electron displacement vectors respectively, which evaluates the
    Coulomb sum for the periodic lattice via the Ewald method.
  """
  rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
  volume = jnp.abs(jnp.linalg.det(lattice))
  # the factor gamma tunes the width of the summands in real / reciprocal space
  # and this value is chosen to optimize the convergence trade-off between the
  # two sums. See CASINO QMC manual.
  root_gamma = 2.4 / volume**0.5
  ordinals = sorted(range(-truncation_limit, truncation_limit + 1), key=abs)
  ordinals = jnp.array(list(itertools.product(ordinals, repeat=2)))
  lat_vectors = jnp.einsum('kj,ij->ik', lattice, ordinals)
  rec_vectors = jnp.einsum('jk,ij->ik', rec, ordinals[1:])
  rec_vec_square = jnp.einsum('ij,ij->i', rec_vectors, rec_vectors)
  rec_vec_norm = jnp.sqrt(rec_vec_square)
  lat_vec_norm = jnp.linalg.norm(lat_vectors[1:], axis=-1)

  def real_space_ewald(separation: jnp.ndarray):
    """Real-space Ewald potential between charges seperated by separation."""
    displacements = jnp.linalg.norm(
        separation - lat_vectors, axis=-1)  # |r - R|
    return jnp.sum(
        jax.scipy.special.erfc(root_gamma * displacements) / displacements)

  def recp_space_ewald(separation: jnp.ndarray):
    """Returns reciprocal-space Ewald potential between charges."""
    return (2 * jnp.pi / volume) * jnp.sum(
        jnp.exp(1.0j * jnp.dot(rec_vectors, separation)) *
        jax.scipy.special.erfc(rec_vec_norm / (2 * root_gamma)) / rec_vec_norm)

  def ewald_sum(separation: jnp.ndarray):
    """Evaluates combined real and reciprocal space Ewald potential."""
    return (real_space_ewald(separation) + recp_space_ewald(separation) -
            2 * (jnp.pi**0.5) / (volume * root_gamma))

  madelung_const = (
      jnp.sum(jax.scipy.special.erfc(root_gamma * lat_vec_norm) / lat_vec_norm)
      - 2 * root_gamma / jnp.pi**0.5)
  madelung_const += (
      (2 * jnp.pi / volume) *
      jnp.sum(jax.scipy.special.erfc(rec_vec_norm / (2 * root_gamma)) / rec_vec_norm) -
      2 * (jnp.pi**0.5) / (volume * root_gamma))

  batch_ewald_sum = jax.vmap(ewald_sum, in_axes=(0,))

  def atom_electron_potential(ae: jnp.ndarray):
    """Evaluates periodic atom-electron potential."""
    nelec = ae.shape[0]
    ae = jnp.reshape(ae, [-1, 2])  # flatten electronxatom axis
    # calculate potential for each ae pair
    ewald = batch_ewald_sum(ae) - madelung_const
    return jnp.sum(-jnp.tile(charges, nelec) * ewald)

  def electron_electron_potential(ee: jnp.ndarray):
    """Evaluates periodic electron-electron potential."""
    nelec = ee.shape[0]
    ee = jnp.reshape(ee, [-1, 2])
    if include_heg_background:
      ewald = batch_ewald_sum(ee)
    else:
      ewald = batch_ewald_sum(ee) - madelung_const
    ewald = jnp.reshape(ewald, [nelec, nelec])
    ewald = ewald.at[jnp.diag_indices(nelec)].set(0.0)
    if include_heg_background:
      return 0.5 * jnp.sum(ewald) + 0.5 * nelec * madelung_const
    else:
      return 0.5 * jnp.sum(ewald)

  # Atom-atom potential
  natom = atoms.shape[0]
  if natom > 1:
    aa = jnp.reshape(atoms, [1, -1, 2]) - jnp.reshape(atoms, [-1, 1, 2])
    aa = jnp.reshape(aa, [-1, 2])
    chargeprods = (charges[..., None] @ charges[..., None].T).flatten()
    ewald = batch_ewald_sum(aa) - madelung_const
    ewald = jnp.reshape(ewald, [natom, natom])
    ewald = ewald.at[jnp.diag_indices(natom)].set(0.0)
    ewald = ewald.flatten()
    atom_atom_potential = 0.5 * jnp.sum(chargeprods * ewald)
  else:
    atom_atom_potential = 0.0

  def potential(ae: jnp.ndarray, ee: jnp.ndarray):
    """Accumulates atom-electron, atom-atom, and electron-electron potential."""
    # Reduce vectors into first unit cell - Ewald summation
    # is only guaranteed to converge close to the origin
    """ Should not be needed if mcmc_pbc is used
    phase_ae = jnp.einsum('il,jkl->jki', rec / (2 * jnp.pi), ae)
    phase_ee = jnp.einsum('il,jkl->jki', rec / (2 * jnp.pi), ee)
    phase_prim_ae = phase_ae % 1
    phase_prim_ee = phase_ee % 1
    prim_ae = jnp.einsum('il,jkl->jki', lattice, phase_prim_ae)
    prim_ee = jnp.einsum('il,jkl->jki', lattice, phase_prim_ee)
    """
    return jnp.real(
        atom_electron_potential(ae) + # prim_ae
        electron_electron_potential(ee) + atom_atom_potential) # prim_ee

  return potential
