# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate Gaussian-Type Orbitals on a grid using Jax."""

import collections
import functools
import itertools
from typing import Any, Mapping, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np
import pyscf.gto
from pyscf.lib import exceptions
from scipy import sparse


def normalize_primitive_weights(basis_list):
  """Correctly handle the primitive weighting and normalization.

  A general basis_list specification for a cGTO of the form
    [L, [alpha_1, w_1], [alpha_2, w_2], ...]
  may not correspond to a basis function that square integrates to 1 unless the
  w_i's are correctly scaled. pyscf.gto.mole.make_bas_env performs the
  necessary scaling of the w_i's, so we execute make_bas_env and read back the
  correctly scaled basis_list from the output.

  Args:
    basis_list: a list of cGTO basis specifications of the form
      [L, [alpha_1, w_1], [alpha_2, w_2], ...]

  Returns:
    a basis_list where all w_i's are correctly scaled to ensure that the cGTOs
    square integrate to 1.
  """
  bas, env = pyscf.gto.mole.make_bas_env(basis_list)
  bas = np.array(bas)
  angl = bas[:, 1]
  start_ptrs = bas[:, 5]
  spec_shape = bas[:, 3:1:-1] + [[1, 0]]
  stop_ptrs = start_ptrs + spec_shape[:, 0] * spec_shape[:, 1]
  basis_list = []
  for l, start, stop, shape in zip(angl, start_ptrs, stop_ptrs, spec_shape):
    basis_list.append([l] + env[start:stop].reshape(shape).T.tolist())
  return basis_list


def full_cart2sph(l: int, reorder_p: bool = False) -> np.ndarray:
  """Computes transform from a complete cartesian representation to spherical.

  Given a complete set of 3**L cartesian derivatives of order L sorted
  alphabetically according to
   > itertools.product('xyz', repeat=L)
  This function returns a [3**L, 2L+1] matrix that converts the cartesian
  representation to 2L+1 components of a spherical harmonic representation.
  We choose to give equal weighting to symmetrically equivalent
  cartesian derivatives e.g. d2/dxdy == d2/dydx in this conversion.

  Args:
    l: the derivative order / spherical harmonic angular momentum
    reorder_p: whether to reorder p orbitals into PySCF order.

  Returns:
    a [3**L, 2L+1] matrix to convert from cartesian to spherical harmonic
    representation.
  """
  # all cartesian combinations
  complete_cart = list(itertools.product(range(3), repeat=l))
  # make a list containing symmetrically unique cartesian combinations
  unique_cart = [x for x in complete_cart if x == tuple(sorted(x))]
  # make an indicator tensor that maps from the complete combinations to the
  # unique combinations
  col_id = [unique_cart.index(tuple(sorted(x))) for x in complete_cart]
  row_id = np.arange(len(complete_cart))
  indicator = np.array(sparse.coo_matrix((np.ones_like(row_id),
                                          (row_id, col_id))).todense())
  # take the average over symmetrically related cartesian derivatives
  indicator = indicator / np.sum(indicator, axis=0, keepdims=True)
  # Add the mapping from the unique_cartesian to the spherical harmonics
  complete_cart2sph = indicator @ pyscf.gto.cart2sph(l)
  if l == 1 and reorder_p:
    return np.array([complete_cart2sph[2],
                     complete_cart2sph[0],
                     complete_cart2sph[1]])
  return complete_cart2sph


def cart2sph(r: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  rho = jnp.linalg.norm(r, axis=-1)
  phi = jnp.arctan2(r[..., 1], r[..., 0])
  theta = jnp.arctan2(jnp.linalg.norm(r[..., :2], axis=-1), r[..., 2])
  return rho, phi, theta


def cartesian_product(r: jnp.ndarray, ell: int) -> jnp.ndarray:
  """Computes all cartesian products of order ell."""
  # i.e. all terms (x^a * y^b * z^c) for a+b+c = ell
  if ell == 0:
    return jnp.array([1.])
  return jnp.prod(jnp.stack(
      jnp.meshgrid(*[r]*ell, indexing='ij'), axis=0).reshape(ell, -1), axis=0)


def solid_harmonic(r: jnp.ndarray, l_max: int) -> jnp.ndarray:
  """Computes all solid harmonics r**ell Y_{ell, m} for all ell <= l_max."""
  r_scalar, phi, theta = cart2sph(r)
  cos_theta = jnp.cos(theta)
  legendre = jss.lpmn_values(l_max, l_max, cos_theta, True)

  m = jnp.arange(l_max + 1)[:, None, None]
  angle = m * phi[None, None, :]
  sign = (-1)**m[1:]
  positive_harmonics = np.sqrt(2) * (legendre[1:] * jnp.cos(angle[1:])) * sign
  zero_harmonics = legendre[0:1]
  negative_harmonics = np.sqrt(2) * (
      (legendre[1:] * jnp.sin(angle[1:])) * sign)[::-1]
  harmonics = jnp.concatenate(
      [negative_harmonics, zero_harmonics, positive_harmonics], axis=0)

  ell = jnp.arange(l_max + 1)[None, :, None]
  return harmonics * (r_scalar[None, None, :]**ell)


def solid_harmonic_from_cart(r: jnp.ndarray, l_max: int) -> jnp.ndarray:
  """Computes all solid harmonics r**ell Y_{ell, m} for all ell <= l_max.

  Intermediate resuts use numerically stable cartesian representations of the
  spherical harmonics.

  Args:
    r: the coordinates at which to evaluate the solid harmonic
    l_max: the maximum order to compute up to.

  Returns:
    a (2*l_max + 1, l_max+1, len(r)) array containing all the solid harmonics
    up to l_max+1.
  """
  harmonics = []
  for ell in range(l_max + 1):
    # TODO(algaunt): There is redundant work in computing cartesian products
    # independent at every ell - can we do it recursively?
    pad = l_max - ell
    cp = jax.vmap(functools.partial(cartesian_product, ell=ell), out_axes=-1)
    # convert the cartesian products to a solid harmonic
    sh = full_cart2sph(ell, True).T @ cp(r)
    # pad the harmonics to the 2*l_max + 1
    sh = jnp.pad(sh, [[pad, pad], [0, 0]])
    harmonics.append(sh)
  return jnp.stack(harmonics, axis=1)  # [L, M, grid]


def grad_solid_harmonic(r: jnp.ndarray, l_max: int) -> jnp.ndarray:
  """Computes all solid harmonics r**ell Y_{ell, m} for all ell <= l_max.

  Intermediate resuts use numerically stable cartesian representations of the
  spherical harmonics.

  Args:
    r: the coordinates at which to evaluate the solid harmonic
    l_max: the maximum order to compute up to.

  Returns:
    a (2*l_max + 1, l_max+1, len(r)) array containing all the solid harmonics
    up to l_max+1.
  """
  def cartesian_sum(r, ell):
    # computes all cartesian sums of order ell
    # i.e. all terms (a*x + b*y + c*z) for a+b+c = ell
    if ell == 0:
      return np.array([0.])
    return np.sum(np.stack(
        np.meshgrid(*[r]*ell, indexing='ij'), axis=0).reshape(ell, -1), axis=0)

  dharmonics = []
  for ell in range(l_max + 1):
    dx = cartesian_sum(np.array([1., 0., 0.]), ell)
    dy = cartesian_sum(np.array([0., 1., 0.]), ell)
    dz = cartesian_sum(np.array([0., 0., 1.]), ell)
    d = np.stack([dx, dy, dz], axis=1)
    cp = jax.vmap(functools.partial(cartesian_product, ell=ell), out_axes=-1)
    dcp = jnp.where(r.T[None, :, :] == 0, 0,
                    d[:, :, None]*cp(r)[:, None, :] / r.T[None, :, :])
    sh = jnp.einsum('cs,cxn->snx', full_cart2sph(ell, True), dcp)
    pad = l_max - ell
    sh = jnp.pad(sh, [[pad, pad], [0, 0], [0, 0]])
    dharmonics.append(sh)

  return jnp.stack(dharmonics, axis=1)  # [L, M, grid, xyz]


def grad_solid_harmonic_by_jacfwd(r: jnp.ndarray, l_max: int) -> jnp.ndarray:
  """Computes the gradient of solid harmonics for all ell <= l_max."""
  dharmonics = []
  for ell in range(l_max + 1):
    # TODO(algaunt) consider use of jvp instead of jacfwd
    dcp = jax.vmap(
        jax.jacfwd(functools.partial(cartesian_product, ell=ell)), out_axes=-1)
    sh = jnp.einsum('cs,cxn->snx', full_cart2sph(ell, True),
                    dcp(r))
    pad = l_max - ell
    sh = jnp.pad(sh, [[pad, pad], [0, 0], [0, 0]])
    dharmonics.append(sh)
  return jnp.stack(dharmonics, axis=1)  # [L, M, grid, xyz]


class Mol:
  """A GTO basis sets evaluator using jax.

  Attributes:
    atom_list: A list of tuples (Z, [x, y, z]) that specifies the atom types (Z)
      and positions (x,y,z) in Bohr for the Mol
    basis_dict: A dictionary of cGTO specifications of the form
      {"Z": [[L, [alpha_1, w_1], [alpha_2, w_2], ...], ...], ...}
    spec: The dict returned by _get_orbital_construction_dict().
  """

  def __init__(self, atom_list, basis_dict):
    self.atom_list = atom_list
    self.basis_dict = basis_dict
    self._spec = self._get_orbital_construction_dict()
    # Needed to provide a concrete value to segment_sum in eval_gto,
    # otherwise tracing + pmap will not work properly
    self._num_segments = self._spec['cshell_id'].max()+1

  @property
  def spec(self) -> Mapping[str, Any]:
    return self._spec

  @classmethod
  def from_pyscf_mol(cls, mol: pyscf.gto.Mole) -> 'Mol':
    """Initialize from a pyscf.gto.Mole."""
    atom_list = pyscf.gto.format_atom(mol.atom, unit=mol.unit)

    def get_basis_for_atom(mol_basis, atom):
      if isinstance(mol_basis, str):
        atom_basis = pyscf.gto.basis.load(mol_basis, atom)
      elif isinstance(mol_basis, dict) and isinstance(mol_basis[atom], str):
        try:
          atom_basis = pyscf.gto.basis.load(mol_basis[atom], atom)
        except exceptions.BasisNotFoundError:
          atom_basis = mol_basis[atom]
      else:
        atom_basis = mol_basis[atom]
      return normalize_primitive_weights(atom_basis)

    basis_dict = {
        atom: get_basis_for_atom(mol.basis, atom)
        for atom in set(list(zip(*atom_list))[0])
    }
    return Mol(atom_list, basis_dict)

  def _get_max_l(self):
    return np.max([basis[0]  # pylint: disable=g-complex-comprehension
                   for atom in self.atom_list
                   for basis in self.basis_dict[atom[0]]])

  def _get_orbital_construction_dict(self):
    """Creates description of all primitive orbitals to be constructed.

    Specifically, this function unravels a contracted basis set description into
    a number of lists that organize construction of pGTOs and contraction to
    cGTOs

    We define 3 concepts:
      pGTO: a primitive GTO of rhe form N exp(-alpha r^2) r^l Y_lm(r)
      cGTO: a contracted GTO which consists of a sum of pGTOs with the same l
        and m but different gaussian precisions a and weightings N
      pShell: a set of all pGTOs consisting of all possible m for orbitals with
        a fixed radial part and angular momentum l
      cShell: like a pShell but made from cGTOs with all m.

    Note that every pGTO requires specification of a radial part and an angular
    part. The radial part is identical for all pGTOs in a pShell.

    The lists are as follows:
      alpha[i]: the precision of the gaussian part of the i^th pShell
      primitive_weights[i]: the weighting N of the i^th pShell including any
        factor needed to normalize the cGTOs to square integrate to 1 after
        contraction.
      atom_id[i]: the index of the atom on which the i^th pShell sits
      cshell_id[i]: the index of the cShell into which the i^th pShell will be
        contracted
      l[i]: the orbital angular momentum of the i^th pShell
      radial_index[i]: the index of the cShell to use for the radial part of
        the i^th cGTO
      angular_index[i]: a tuple containing of the orbital angular momentum, the
        azimuthal quantum number and the atom_id of the i^th cGTO

    Additionally the dictionary contains
      atom_centres[i]: the x,y,z, coordinate of the i^th atom
      l_max: the largest orbital angular momentum in the basis

    Returns:
      A dictionary of the construction lists described above
    """
    construction_spec = collections.defaultdict(list)
    cshell_id = 0
    for atom_id, atom in enumerate(self.atom_list):
      for basis in self.basis_dict[atom[0]]:
        l = basis[0]
        n_weights = len(basis[1]) - 1
        for w in range(n_weights):
          for info in basis[1:]:
            # Only store an uncontracted exponent if it is actually used
            # in a basis set.
            if abs(info[w+1]) > 1E-12:
              construction_spec['alpha'].append(info[0])
              construction_spec['primitive_weights'].append(info[w+1])
              construction_spec['atom_id'].append(atom_id)
              construction_spec['cshell_id'].append(cshell_id)
              construction_spec['l'].append(l)
          construction_spec['radial_index'] += [
              cshell_id for _ in range(2 * l + 1)
          ]
          ms = [1, -1, 0] if l == 1 else range(-l, l +
                                               1)  # pyscf reorders p orbitals
          for m in ms:
            construction_spec['angular_index'].append((l, m, atom_id))
          cshell_id += 1
    construction_spec['atom_centres'] = np.array(list(zip(*self.atom_list))[1])
    for k, v in construction_spec.items():
      construction_spec[k] = np.array(v)
    return construction_spec

  def eval_gto(self, coords: jnp.ndarray) -> jnp.ndarray:
    r"""Computes all gtos on the grid of coords.

    A primitive GTO consists of the product of a gaussian radial part and a
    solid harmonic angular part

      N exp(-alpha r^2) r^l Y_lm(r)
      \--------------/  \--------/
        radial part     angular part

    The radial part is the same for all elements in a cShell, so we first
    construct all radial part for all cShells and then join these with the
    angular parts as dictated by the lists in self._spec.

    Args:
      coords: a [G, 3] array containing the xyz coords at which to evaluate the
        GTOs

    Returns:
      A [G, CGTO] array of the evaluated GTOs.
    """
    # shape annotations:
    #   G = len(coords),
    #   A = number of atoms
    #   L = max_l
    #   PGTO = number of primitive GTOs
    #   CGTO = mol.nao_nr() (number of contrated GTOs)
    #   CSHELL = mol.nbas = number of contracted shells

    # construct copies of the grid centred on each atom [G, A, 3]
    dr = coords[:, None, :] - self._spec['atom_centres']
    flat_dr = dr.reshape(-1, 3)
    # construct all solid harmonics [2L+1, L, (G*A)]
    max_l = self._get_max_l()
    sh = solid_harmonic(flat_dr, max_l)
    sh = sh.reshape(sh.shape[0], sh.shape[1], dr.shape[0], dr.shape[1])

    # construct the radial part
    r_sqr = jnp.linalg.norm(dr, axis=-1)**2  # [G, A]
    g = jnp.exp(-self._spec['alpha'][None, :] * r_sqr[:, self._spec['atom_id']]
               ) * self._spec['primitive_weights'][None, :]  # [G, PGTO]

    # contract the primitives
    g = jax.ops.segment_sum(  # [CSHELL, G]
        g.T, self._spec['cshell_id'], num_segments=self._num_segments)
    radial_part = g[self._spec['radial_index']]  # [CGTO, G]

    angular_part = sh[self._spec['angular_index'][:, 1] + max_l,
                      self._spec['angular_index'][:, 0], :,
                      self._spec['angular_index'][:, 2]]

    return (angular_part * radial_part).T  # [G, CGTO]
