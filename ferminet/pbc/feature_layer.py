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

"""Feature layer for periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

from typing import Optional, Tuple

import chex
from ferminet import networks
import jax.numpy as jnp


def periodic_norm(metric: jnp.ndarray, scaled_r: jnp.ndarray) -> jnp.ndarray:
  """Returns the periodic norm of a set of vectors.

  Args:
    metric: metric tensor in fractional coordinate system, A.T A, where A is the
      lattice vectors.
    scaled_r: vectors in fractional coordinates of the lattice cell, with
      trailing dimension ndim, to compute the periodic norm of.
  """
  chex.assert_rank(metric, expected_ranks=2)
  a = (1 - jnp.cos(2 * jnp.pi * scaled_r))
  b = jnp.sin(2 * jnp.pi * scaled_r)
  cos_term = jnp.einsum('...m,mn,...n->...', a, metric, a)
  sin_term = jnp.einsum('...m,mn,...n->...', b, metric, b)
  return (1 / (2 * jnp.pi)) * jnp.sqrt(cos_term + sin_term)


def make_pbc_feature_layer(
    natoms: Optional[int] = None,
    nspins: Optional[Tuple[int, ...]] = None,
    ndim: int = 3,
    rescale_inputs: bool = False,
    lattice: Optional[jnp.ndarray] = None,
    include_r_ae: bool = True,
) -> networks.FeatureLayer:
  """Returns the init and apply functions for periodic features.

  Args:
      natoms: number of atoms.
      nspins: tuple of the number of spin-up and spin-down electrons.
      ndim: dimension of the system.
      rescale_inputs: If true, rescales r_ae for stability. Note that unlike in
        the OBC case, we do not rescale r_ee as well.
      lattice: Matrix whose columns are the primitive lattice vectors of the
        system, shape (ndim, ndim).
      include_r_ae: Flag to enable electron-atom distance features. Set to False
        to avoid cusps with ghost atoms in, e.g., homogeneous electron gas.
  """

  del nspins

  if lattice is None:
    lattice = jnp.eye(ndim)

  # Calculate reciprocal vectors, factor 2pi omitted
  reciprocal_vecs = jnp.linalg.inv(lattice)
  lattice_metric = lattice.T @ lattice

  def init() -> Tuple[Tuple[int, int], networks.Param]:
    if include_r_ae:
      return (natoms * (2 * ndim + 1), 2 * ndim + 1), {}
    else:
      return (natoms * (2 * ndim), 2 * ndim + 1), {}

  def apply(ae, r_ae, ee, r_ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # One e features in phase coordinates, (s_ae)_i = k_i . ae
    s_ae = jnp.einsum('il,jkl->jki', reciprocal_vecs, ae)
    # Two e features in phase coordinates
    s_ee = jnp.einsum('il,jkl->jki', reciprocal_vecs, ee)
    # Periodized features
    ae = jnp.concatenate(
        (jnp.sin(2 * jnp.pi * s_ae), jnp.cos(2 * jnp.pi * s_ae)), axis=-1)
    ee = jnp.concatenate(
        (jnp.sin(2 * jnp.pi * s_ee), jnp.cos(2 * jnp.pi * s_ee)), axis=-1)
    # Distance features defined on orthonormal projections
    r_ae = periodic_norm(lattice_metric, s_ae)
    if rescale_inputs:
      r_ae = jnp.log(1 + r_ae)
    # Don't take gradients through |0|
    n = ee.shape[0]
    s_ee += jnp.eye(n)[..., None]
    r_ee = periodic_norm(lattice_metric, s_ee) * (1.0 - jnp.eye(n))

    if include_r_ae:
      ae_features = jnp.concatenate((r_ae[..., None], ae), axis=2)
    else:
      ae_features = ae
    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
    ee_features = jnp.concatenate((r_ee[..., None], ee), axis=2)
    return ae_features, ee_features

  return networks.FeatureLayer(init=init, apply=apply)
