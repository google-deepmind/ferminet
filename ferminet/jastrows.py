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

"""Multiplicative Jastrow factors."""

import enum
from typing import Any, Callable, Iterable, Mapping, Union, Optional

import jax.numpy as jnp

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
  """Available multiplicative Jastrow factors."""

  NONE = enum.auto()
  SIMPLE_EE = enum.auto()


def make_periodic_r_ee(lattice: jnp.ndarray):
  """Util function for transforming r_ee into the periodic version.
  Some of this could be refactored with ferminet.pbc.feature_layer

  Args:
      lattice: Matrix whose columns are the primitive lattice vectors of the
        system, shape (ndim, ndim).
  """

  # Calculate reciprocal vectors, factor 2pi omitted
  reciprocal_vecs = jnp.linalg.inv(lattice)
  lattice_metric = lattice.T @ lattice

  def apply(ee: jnp.ndarray):
    s_ee = jnp.einsum('il,jkl->jki', reciprocal_vecs, ee)

    n = ee.shape[0]
    s_ee += jnp.eye(n)[..., None]

    a = (1 - jnp.cos(2 * jnp.pi * s_ee))
    b = jnp.sin(2 * jnp.pi * s_ee)
    cos_term = jnp.einsum('...m,mn,...n->...', a, lattice_metric, a)
    sin_term = jnp.einsum('...m,mn,...n->...', b, lattice_metric, b)
    periodic_r_ee = (1 / (2 * jnp.pi)) * jnp.sqrt(cos_term + sin_term)

    periodic_r_ee = periodic_r_ee * (1.0 - jnp.eye(n))
    return periodic_r_ee[..., None]
  
  return apply


def _jastrow_ee(
    r_ee: jnp.ndarray,
    params: ParamTree,
    nspins: tuple[int, int],
    ndim: int,
    jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
  """Jastrow factor for electron-electron cusps."""
  r_ees = [
      jnp.split(r, nspins[0:1], axis=1)
      for r in jnp.split(r_ee, nspins[0:1], axis=0)
  ]
  r_ees_parallel = jnp.concatenate([
      r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
      r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
  ])

  if r_ees_parallel.shape[0] > 0:
    jastrow_ee_par = jnp.sum(
        jastrow_fun(r_ees_parallel, 1 / (ndim + 1), params['ee_par'])
    )
  else:
    jastrow_ee_par = jnp.asarray(0.0)

  if r_ees[0][1].shape[0] > 0:
    jastrow_ee_anti = jnp.sum(
        jastrow_fun(r_ees[0][1], 1 / (ndim - 1), params['ee_anti']))
  else:
    jastrow_ee_anti = jnp.asarray(0.0)

  return jastrow_ee_anti + jastrow_ee_par


def make_simple_ee_jastrow(lattice: Optional[jnp.ndarray] = None, ndim: int = 3):
  """Creates a Jastrow factor for electron-electron cusps."""

  # If working in PBC, use periodic distance for the Jastrow
  if lattice is not None:
    norm = make_periodic_r_ee(lattice)
  else:
    norm = lambda x: jnp.linalg.norm(x, axis = -1, keepdims = True)

  def simple_ee_cusp_fun(
      r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    """Jastrow function satisfying electron cusp condition."""
    return -(cusp * alpha**2) / (alpha + r)

  def init() -> Mapping[str, jnp.ndarray]:
    params = {}
    params['ee_par'] = jnp.ones(
        shape=1,
    )
    params['ee_anti'] = jnp.ones(
        shape=1,
    )
    return params

  def apply(
      ee: jnp.ndarray,
      params: ParamTree,
      nspins: tuple[int, int],
  ) -> jnp.ndarray:
    """Jastrow factor for electron-electron cusps."""
    r_ee = norm(ee)
    return _jastrow_ee(r_ee, params, nspins, ndim, jastrow_fun=simple_ee_cusp_fun)

  return init, apply


def get_jastrow(
    jastrow: JastrowType, 
    lattice: Optional[jnp.ndarray] = None, 
    ndim: int = 3):
  jastrow_init, jastrow_apply = None, None
  if jastrow == JastrowType.SIMPLE_EE:
    jastrow_init, jastrow_apply = make_simple_ee_jastrow(lattice, ndim)
  elif jastrow != JastrowType.NONE:
    raise ValueError(f'Unknown Jastrow Factor type: {jastrow}')

  return jastrow_init, jastrow_apply
