# Copyright 2026 DeepMind Technologies Limited.
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

"""2D and 3D Ewald summation of Coulomb Hamiltonian in periodic boundary
conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

import itertools
from typing import Callable

import jax
import jax.numpy as jnp

from ferminet import mcmc


def make_ewald_potential(
    lattice: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    truncation_limit: int = 5,
    ndim: int = 3,
) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
  """Creates a function to evaluate infinite Coulomb sum for periodic lattice.

  Electrons have charge -1. A uniform background neutralizing the total charge
  of the cell is always included, so it has no effect for charge-neutral cells
  and gives the jellium background for the homogeneous electron gas.

  Args:
    lattice: Shape (ndim, ndim). Matrix whose columns are the primitive lattice
      vectors.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
    truncation_limit: Integer. Half side length of cube of nearest neighbours
      to primitive cell which are summed over in evaluation of Ewald sum.
      Must be large enough to achieve convergence for the real and reciprocal
      space sums.
    ndim: Integer. Number of spatial dimensions. Must be 2 or 3.

  Returns:
    Callable with signature f(ae, ee), where (ae, ee) are atom-electon and
    electron-electron displacement vectors respectively, which evaluates the
    Coulomb sum for the periodic lattice via the Ewald method.
  """
  if ndim not in (2, 3):
    raise ValueError(f'ndim must be 2 or 3, got {ndim}.')
  rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
  volume = jnp.abs(jnp.linalg.det(lattice))
  # the factor gamma tunes the width of the summands in real / reciprocal space
  # and this value is chosen to optimize the convergence trade-off between the
  # two sums. See CASINO QMC manual.
  if ndim == 2:
    root_gamma = 2.4 / volume**0.5
  else:
    root_gamma = 2.8 / volume**(1 / 3)
  ordinals = sorted(range(-truncation_limit, truncation_limit + 1), key=abs)
  ordinals = jnp.array(list(itertools.product(ordinals, repeat=ndim)))
  lat_vectors = jnp.einsum('kj,ij->ik', lattice, ordinals)
  rec_vectors = jnp.einsum('jk,ij->ik', rec, ordinals[1:])
  rec_vec_square = jnp.einsum('ij,ij->i', rec_vectors, rec_vectors)
  rec_vec_norm = jnp.sqrt(rec_vec_square)
  lat_vec_norm = jnp.linalg.norm(lat_vectors[1:], axis=-1)

  if ndim == 2:
    rec_weights = (2 * jnp.pi / volume) * jax.scipy.special.erfc(
        rec_vec_norm / (2 * root_gamma)) / rec_vec_norm
    g0_term = 2 * (jnp.pi**0.5) / (volume * root_gamma)
  else:
    rec_weights = (4 * jnp.pi / volume) * jnp.exp(
        -rec_vec_square / (4 * root_gamma**2)) / rec_vec_square
    g0_term = jnp.pi / (volume * root_gamma**2)

  def real_space_ewald(separation: jnp.ndarray):
    """Real-space Ewald potential between charges seperated by separation."""
    displacements = jnp.linalg.norm(
        separation - lat_vectors, axis=-1)  # |r - R|
    return jnp.sum(
        jax.scipy.special.erfc(root_gamma * displacements) / displacements)

  def recp_space_ewald(separation: jnp.ndarray):
    """Returns reciprocal-space Ewald potential between charges."""
    return jnp.sum(
        jnp.exp(1.0j * jnp.dot(rec_vectors, separation)) * rec_weights)

  def ewald_sum(separation: jnp.ndarray):
    """Evaluates combined real and reciprocal space Ewald potential."""
    return (real_space_ewald(separation) + recp_space_ewald(separation) -
            g0_term)

  madelung_const = (
      jnp.sum(jax.scipy.special.erfc(root_gamma * lat_vec_norm) / lat_vec_norm)
      - 2 * root_gamma / jnp.pi**0.5)
  madelung_const += jnp.sum(rec_weights) - g0_term

  batch_ewald_sum = jax.vmap(ewald_sum, in_axes=(0,))

  def atom_electron_potential(ae: jnp.ndarray):
    """Evaluates periodic atom-electron potential."""
    nelec = ae.shape[0]
    ae = jnp.reshape(ae, [-1, ndim])  # flatten electronxatom axis
    # calculate potential for each ae pair
    ewald = batch_ewald_sum(ae)
    return jnp.sum(-jnp.tile(charges, nelec) * ewald)

  def electron_electron_potential(ee: jnp.ndarray):
    """Evaluates periodic electron-electron potential."""
    nelec = ee.shape[0]
    ee = jnp.reshape(ee, [-1, ndim])
    ewald = batch_ewald_sum(ee)
    ewald = jnp.reshape(ewald, [nelec, nelec])
    ewald = ewald.at[jnp.diag_indices(nelec)].set(0.0)
    return 0.5 * jnp.sum(ewald) + 0.5 * nelec * madelung_const

  # Atom-atom potential
  natom = atoms.shape[0]
  if natom > 1:
    aa = jnp.reshape(atoms, [1, -1, ndim]) - jnp.reshape(atoms, [-1, 1, ndim])
    aa = jnp.reshape(aa, [-1, ndim])
    chargeprods = (charges[..., None] @ charges[..., None].T).flatten()
    ewald = batch_ewald_sum(aa)
    ewald = jnp.reshape(ewald, [natom, natom])
    ewald = ewald.at[jnp.diag_indices(natom)].set(0.0)
    ewald = ewald.flatten()
    atom_atom_potential = 0.5 * jnp.sum(chargeprods * ewald)
  else:
    atom_atom_potential = 0.0
  atom_atom_potential += 0.5 * jnp.sum(charges**2) * madelung_const

  def potential(ae: jnp.ndarray, ee: jnp.ndarray):
    """Accumulates atom-electron, atom-atom, and electron-electron potential."""
    # Reduce vectors into first unit cell - Ewald summation
    # is only guaranteed to converge close to the origin
    ae_shape = ae.shape
    ee_shape = ee.shape
    ae = mcmc.map_to_simulation_cell(ae, lattice, rec, ndim).reshape(ae_shape)
    ee = mcmc.map_to_simulation_cell(ee, lattice, rec, ndim).reshape(ee_shape)
    return jnp.real(
        atom_electron_potential(ae) +
        electron_electron_potential(ee) + atom_atom_potential)

  return potential