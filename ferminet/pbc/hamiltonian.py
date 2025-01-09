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

"""Ewald summation of Coulomb Hamiltonian in periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

import itertools
from typing import Callable, Optional, Sequence, Tuple

import chex
from ferminet import hamiltonian
from ferminet import networks
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
    lattice: Shape (3, 3). Matrix whose columns are the primitive lattice
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
  gamma = (2.8 / volume**(1 / 3))**2
  ordinals = sorted(range(-truncation_limit, truncation_limit + 1), key=abs)
  ordinals = jnp.array(list(itertools.product(ordinals, repeat=3)))
  lat_vectors = jnp.einsum('kj,ij->ik', lattice, ordinals)
  rec_vectors = jnp.einsum('jk,ij->ik', rec, ordinals[1:])
  rec_vec_square = jnp.einsum('ij,ij->i', rec_vectors, rec_vectors)
  lat_vec_norm = jnp.linalg.norm(lat_vectors[1:], axis=-1)

  def real_space_ewald(separation: jnp.ndarray):
    """Real-space Ewald potential between charges seperated by separation."""
    displacements = jnp.linalg.norm(
        separation - lat_vectors, axis=-1)  # |r - R|
    return jnp.sum(
        jax.scipy.special.erfc(gamma**0.5 * displacements) / displacements)

  def recp_space_ewald(separation: jnp.ndarray):
    """Returns reciprocal-space Ewald potential between charges."""
    return (4 * jnp.pi / volume) * jnp.sum(
        jnp.exp(1.0j * jnp.dot(rec_vectors, separation)) *
        jnp.exp(-rec_vec_square / (4 * gamma)) / rec_vec_square)

  def ewald_sum(separation: jnp.ndarray):
    """Evaluates combined real and reciprocal space Ewald potential."""
    return (real_space_ewald(separation) + recp_space_ewald(separation) -
            jnp.pi / (volume * gamma))

  madelung_const = (
      jnp.sum(jax.scipy.special.erfc(gamma**0.5 * lat_vec_norm) / lat_vec_norm)
      - 2 * gamma**0.5 / jnp.pi**0.5)
  madelung_const += (
      (4 * jnp.pi / volume) *
      jnp.sum(jnp.exp(-rec_vec_square / (4 * gamma)) / rec_vec_square) -
      jnp.pi / (volume * gamma))

  batch_ewald_sum = jax.vmap(ewald_sum, in_axes=(0,))

  def atom_electron_potential(ae: jnp.ndarray):
    """Evaluates periodic atom-electron potential."""
    nelec = ae.shape[0]
    ae = jnp.reshape(ae, [-1, 3])  # flatten electronxatom axis
    # calculate potential for each ae pair
    ewald = batch_ewald_sum(ae) - madelung_const
    return jnp.sum(-jnp.tile(charges, nelec) * ewald)

  def electron_electron_potential(ee: jnp.ndarray):
    """Evaluates periodic electron-electron potential."""
    nelec = ee.shape[0]
    ee = jnp.reshape(ee, [-1, 3])
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
    aa = jnp.reshape(atoms, [1, -1, 3]) - jnp.reshape(atoms, [-1, 1, 3])
    aa = jnp.reshape(aa, [-1, 3])
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
    phase_ae = jnp.einsum('il,jkl->jki', rec / (2 * jnp.pi), ae)
    phase_ee = jnp.einsum('il,jkl->jki', rec / (2 * jnp.pi), ee)
    phase_prim_ae = phase_ae % 1
    phase_prim_ee = phase_ee % 1
    prim_ae = jnp.einsum('il,jkl->jki', lattice, phase_prim_ae)
    prim_ee = jnp.einsum('il,jkl->jki', lattice, phase_prim_ee)
    return jnp.real(
        atom_electron_potential(prim_ae) +
        electron_electron_potential(prim_ee) + atom_atom_potential)

  return potential


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',
    states: int = 0,
    lattice: Optional[jnp.ndarray] = None,
    heg: bool = True,
    convergence_radius: int = 5,
) -> hamiltonian.LocalEnergy:
  """Creates the local energy function in periodic boundary conditions.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian
    states: Number of excited states to compute. Not implemented, only present
      for consistency of calling convention.
    lattice: Shape (ndim, ndim). Matrix of lattice vectors. Default: identity
      matrix.
    heg: bool. Flag to enable features specific to the electron gas.
    convergence_radius: int. Radius of cluster summed over by Ewald sums.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  if states:
    raise NotImplementedError('Excited states not implemented with PBC.')
  del nspins
  if lattice is None:
    lattice = jnp.eye(3)

  ke = hamiltonian.local_kinetic_energy(f,
                                        use_scan=use_scan,
                                        complex_output=complex_output,
                                        laplacian_method=laplacian_method)

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
    potential_energy = make_ewald_potential(
        lattice, data.atoms, charges, convergence_radius, heg
    )
    ae, ee, _, _ = networks.construct_input_features(
        data.positions, data.atoms)
    potential = potential_energy(ae, ee)
    kinetic = ke(params, data)
    return potential + kinetic, None

  return _e_l
