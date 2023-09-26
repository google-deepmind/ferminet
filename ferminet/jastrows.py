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
from typing import Any, Callable, Iterable, Mapping, Union

import jax.numpy as jnp

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
  """Available multiplicative Jastrow factors."""

  NONE = enum.auto()
  SIMPLE_EE = enum.auto()


def _jastrow_ee(
    r_ee: jnp.ndarray,
    params: ParamTree,
    nspins: tuple[int, int],
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
        jastrow_fun(r_ees_parallel, 0.25, params['ee_par'])
    )
  else:
    jastrow_ee_par = jnp.asarray(0.0)

  if r_ees[0][1].shape[0] > 0:
    jastrow_ee_anti = jnp.sum(jastrow_fun(r_ees[0][1], 0.5, params['ee_anti']))
  else:
    jastrow_ee_anti = jnp.asarray(0.0)

  return jastrow_ee_anti + jastrow_ee_par


def make_simple_ee_jastrow() -> ...:
  """Creates a Jastrow factor for electron-electron cusps."""

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
      r_ee: jnp.ndarray,
      params: ParamTree,
      nspins: tuple[int, int],
  ) -> jnp.ndarray:
    """Jastrow factor for electron-electron cusps."""
    return _jastrow_ee(r_ee, params, nspins, jastrow_fun=simple_ee_cusp_fun)

  return init, apply


def get_jastrow(jastrow: JastrowType) -> ...:
  jastrow_init, jastrow_apply = None, None
  if jastrow == JastrowType.SIMPLE_EE:
    jastrow_init, jastrow_apply = make_simple_ee_jastrow()
  elif jastrow != JastrowType.NONE:
    raise ValueError(f'Unknown Jastrow Factor type: {jastrow}')

  return jastrow_init, jastrow_apply
