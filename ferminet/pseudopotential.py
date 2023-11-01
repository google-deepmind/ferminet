# Copyright 2023 DeepMind Technologies Limited
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

"""Evaluates the pseudopotential Hamiltonian on a wavefunction."""

from typing import Sequence

from ferminet.utils import elements
from ferminet.utils import pseudopotential as pp_utils
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import pyscf


def vmap_sum(f, *vmap_args, **vmap_kwargs):
  """Decorator for nested vmaps where the result should be summed anyways.

  Performs intermediate sums to reduce size of final array. Note that all axes
  of the result are summed over.

  Args:
    f: function to vmap and reduce.
    *vmap_args: args to pass to vmap.
    **vmap_kwargs: kwargs to pass to vmap.

  Returns:
    callable which applies f and sums over the vmapped axis.
  """
  vmapped_f = jax.vmap(f, *vmap_args, **vmap_kwargs)

  def vmap_summed_f(*args, **kwargs):
    return jnp.sum(vmapped_f(*args, **kwargs), axis=0)

  return vmap_summed_f


def construct_align(key, v):
  """Makes rotation matrix to align a z axis along v."""
  # e1, e2, e3 = jnp.eye(3) # for debugging

  if v.shape != (3,):
    raise ValueError('v must be a three-dimensional vector')
  ep3 = v / jnp.linalg.norm(v)
  ep2 = jnp.cross(ep3, jax.random.normal(key, shape=(3,)))
  ep2 /= jnp.linalg.norm(ep2)

  # all normalized
  ep1 = jnp.cross(ep2, ep3)
  return jnp.array([ep1, ep2, ep3]).T


def project_legendre(
    direction,
    signed_f,
    params,
    data,
    out_denom,
    electron_distance_i,
    electron_vector_i,
    index,
    atom_vector,
    angular_momentum,
    complex_output
):
  """Projects the Legrende polynomials."""
  cos = direction @ electron_vector_i / electron_distance_i
  leg = pp_utils.eval_leg(cos, angular_momentum)
  e_prime = electron_distance_i * direction
  e = e_prime + atom_vector

  out_num = signed_f(
      params,
      data.positions.reshape(-1, 3).at[index].set(e).ravel(),
      data.spins,
      data.atoms,
      data.charges,
  )
  if complex_output:
    f_ratio = jnp.exp((out_num[1] - out_denom[1])
                      + 1.j * (out_num[0] - out_denom[0]))
  else:
    f_ratio = out_num[0] / out_denom[0] * jnp.exp(out_num[1] - out_denom[1])

  return jnp.squeeze(leg * f_ratio)


def make_spherical_integral(quad_degree):
  """Creates callable for evaluating an integral over a spherical quadrature."""

  if quad_degree != 4:
    raise RuntimeError('quad_degree = 4 is the only implemented quadrature')
  # This matches (up to rotation and permutation) quadpy.u3.get_good_scheme(4)
  # and is the quadrature used in the ByteDance pseudopotential paper etc.
  n_points = 12
  weights = jnp.ones(n_points) / n_points
  spherical_points = [[0, 0], [np.pi, 0]]
  spherical_points += [[np.arctan(2), 2 * np.pi * i / 5] for i in range(1, 6)]
  spherical_points += [
      [np.pi - np.arctan(2), np.pi / 5 * (2 * i - 11)] for i in range(6, 11)
  ]
  theta, phi = zip(*spherical_points)
  points = jnp.stack([
      np.cos(phi) * np.sin(theta),
      np.sin(phi) * np.sin(theta),
      np.cos(theta),
  ], axis=1)

  def spherical_integral(
      key,
      signed_f,
      params,
      data,
      electron_atom_distance,
      electron_atom_vector,
      electron_index,
      atom_position,
      angular_momentum,
      complex_output
  ):

    key, subkey = jax.random.split(key)
    z = jax.random.normal(subkey, shape=(3,))
    rot_mat = construct_align(key, z)
    aligned_points = (rot_mat @ points.T).T
    out_denom = signed_f(params,
                         data.positions,
                         data.spins,
                         data.atoms,
                         data.charges)

    def _body_fun(i, val):
      return val + weights[i] * project_legendre(
          aligned_points[i],
          signed_f,
          params,
          data,
          out_denom,
          electron_atom_distance,
          electron_atom_vector,
          electron_index,
          atom_position,
          angular_momentum,
          complex_output
      )

    init_loop = jnp.zeros_like(out_denom[1])
    if complex_output:
      if out_denom[1].dtype == jnp.float32:
        init_loop = init_loop.astype(jnp.complex64)
      elif out_denom[1].dtype == jnp.float64:
        init_loop = init_loop.astype(jnp.complex128)
    return (2 * angular_momentum + 1) * lax.fori_loop(
        0, n_points, _body_fun, init_loop)

  return spherical_integral


def pp_loc(electron_atom_distance, r_grid, v_grid):
  """Returns the local pseudopotential energy, vmapped over atoms.

  Vmapped axis of input arguments are denoted with underscores.

  Args:
    electron_atom_distance: radial distances to all the electrons. Shape
      (nelectron, _n_atom_).
    r_grid: interpolation grid. Shape (n_grid,).
    v_grid: interpolation values potential = V*r. Shape (_n_atom_, n_grid,).
  """
  return jnp.sum(jnp.interp(electron_atom_distance, xp=r_grid, fp=v_grid))


def make_local_pseudopotential(r_grid, v_grid_loc, ecp_mask):
  # vmap over atoms
  vmap_pp_loc = vmap_sum(pp_loc, in_axes=(1, None, 0))

  def electron_atom_local_pseudopotential(r_ae):
    # mask out non-pseudized atoms
    r_ae_ecp = r_ae[:, ecp_mask, 0]
    return vmap_pp_loc(r_ae_ecp, r_grid, v_grid_loc)

  return electron_atom_local_pseudopotential


def make_nonlocal_pseudopotential(charges, r_grid, v_grid_nonloc, ecp_mask,
                                  effective_charges, quad_degree,
                                  complex_output):
  """Creates callable for evaluating the non-local pseudopotential."""
  del charges, effective_charges  # unused
  spherical_integral = make_spherical_integral(quad_degree)

  # pp_nonloc args:
  #  0:key, (nelec, nchan, :)
  #  1:signed_f (None)
  #  2:params (None)
  #  3:data (None)
  #  4:spherical_integral (None)
  #  5:electron_atom_distance (nelec)
  #  6:electron_atom_vector (nelec, :)
  #  7:electron_index (nelec)
  #  8:atom_position (nelec, :)
  #  9:angular_momentum (nchan)
  #  10:r_grid (:)
  #  11:v_grid (nelec, nchan, :)

  def pp_nonloc(
      key,
      signed_f,
      params,
      data,
      spherical_integral,
      electron_atom_distance,
      electron_atom_vector,
      electron_index,
      atom_position,
      angular_momentum,
      r_grid,
      v_grid,
  ):
    """Evaluates nonlocal pseudopotential.

    Pseudopotential is vmapped over angular momentum, atoms and electrons.
    Vmapped axis of input arguments are denoted with underscores.

    Args:
      key : PRNGKey, shape (_n_l_, _n_atom_, _n_electron_, 2)
      signed_f : sign/log magnitude of the wavefunction.
      params: network parameters.
      data: MCMC configuration.
      spherical_integral: function to evaluate a spherical integral using
        numerical quadrature.
      electron_atom_distance : magnitude of electron_atom_vector. Shape
        (_nelectron_, _natom_).
      electron_atom_vector : vector from atom to electron.  Shape (_nelectron_,
        _natom_, ndim).
      electron_index: index of electron to evaluate pseudopotential for. Shape
        (_nelectron_).
      atom_position: atom position of the atom closest to each electron. Shape
        (_nelectron_, ndim,)
      angular_momentum: angular momentum channel. Shape (_n_l_)
      r_grid: Shape (n_grid,)
      v_grid: Shape (n_grid, _n_l_)

    Returns:
      e_nonloc: Shape (n_atom, n_l).
    """

    integral = spherical_integral(
        key,
        signed_f,
        params,
        data,
        electron_atom_distance,
        electron_atom_vector,
        electron_index,
        atom_position,
        angular_momentum,
        complex_output
    )

    potential_radial_value = jnp.interp(
        electron_atom_distance, xp=r_grid, fp=v_grid)

    return potential_radial_value * integral

  # vmap over electrons
  vmap_pp_nonloc = vmap_sum(
      pp_nonloc,
      in_axes=(0, None, None, None, None, 0, 0, 0, 0, None, None, 0))

  # vmap over angular momentum
  vmap_pp_nonloc = vmap_sum(
      vmap_pp_nonloc,
      in_axes=(1, None, None, None, None, None, None, None, None, 0, None, 1))

  def electron_atom_nonlocal_pseudopotential(
      key, signed_f, params, data, ae, r_ae):
    """Evaluates electron-atom contribution to non-local pseudopotential.

    Args:
      key: Jax PRNGKey for randomized evaluation of spherical integrals.
      signed_f : sign/log magnitude of the wavefunction.
      params: network parameters.
      data: MCMC configuration.
      ae: atom-electron vector. Shape (nelectron, natom, ndim).
      r_ae: atom-electron vector. Shape (nelectron, natom, 1).

    Returns:
      electron-atom contribution to the non-local pseudopotential.
    """
    n_elec = r_ae.shape[0]
    n_nonloc = v_grid_nonloc.shape[1]

    # drop non-ecp atoms
    r_ae_ecp = r_ae[:, ecp_mask, 0]
    ae_ecp = ae[:, ecp_mask]

    # drop all but closest ecp atom
    closest_atom = jnp.argmin(r_ae_ecp, axis=1)
    # keep first index intact
    all_electrons = jnp.arange(n_elec)

    r_ae_ecp_closest = r_ae_ecp[all_electrons, closest_atom]
    ae_ecp_closest = ae_ecp[all_electrons, closest_atom, :]

    atoms_ecp_closest = data.atoms[ecp_mask][closest_atom]

    v_grid_nonloc_closest = v_grid_nonloc[closest_atom]

    keys = jax.random.split(key, n_nonloc * n_elec).reshape(n_elec, n_nonloc, 2)

    return vmap_pp_nonloc(
        keys,
        signed_f,
        params,
        data,
        spherical_integral,
        r_ae_ecp_closest,
        ae_ecp_closest,
        jnp.arange(n_elec),
        atoms_ecp_closest,
        jnp.arange(n_nonloc),
        r_grid,
        v_grid_nonloc_closest,
    )

  return electron_atom_nonlocal_pseudopotential


def make_pp_potential(
    charges: jnp.ndarray,
    symbols: Sequence[str],
    quad_degree: int = 4,
    ecp: str = 'ccecp',
    complex_output: bool = False,
) -> ...:
  """Constructs evaluation of potential due to pseudopotential."""
  ecp_nwchem_fmt = {
      elements.SYMBOLS[symb].atomic_number:
      pyscf.gto.basis.load_ecp(ecp, symb) for symb in symbols
  }

  n_cores, v_grid_dict, r_grid, n_channels = pp_utils.eval_ecp_on_grid(
      ecp_nwchem_fmt)
  # residual atomic charge
  effective_charges = jnp.asarray(
      [z - n_cores.get(z, 0) for z in charges.tolist()]
  )

  # mask to separate pseudo atoms from regular atoms
  ecp_mask = np.abs(np.asarray(charges) - effective_charges) > 2.e-6
  # construct v_grids
  n_ecp = ecp_mask.sum()
  n_grid = r_grid.size
  v_grid = jnp.zeros((n_ecp, n_channels, n_grid))

  for i, z in enumerate(charges[ecp_mask].tolist()):
    v_grid = v_grid.at[i].set(v_grid_dict[z])

  v_grid_nonloc, v_grid_loc = jnp.split(v_grid, (n_channels - 1,), axis=1)
  v_grid_loc = v_grid_loc.reshape(-1, n_grid)

  pp_local = make_local_pseudopotential(r_grid, v_grid_loc, ecp_mask)
  pp_nonlocal = make_nonlocal_pseudopotential(charges, r_grid, v_grid_nonloc,
                                              ecp_mask, effective_charges,
                                              quad_degree, complex_output)

  return effective_charges, pp_local, pp_nonlocal
