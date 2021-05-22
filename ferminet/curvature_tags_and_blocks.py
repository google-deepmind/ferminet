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

"""Curvature blocks for FermiNet."""
import functools
from typing import Optional, Mapping, Union
import jax
import jax.numpy as jnp

from kfac_ferminet_alpha import curvature_blocks as blocks
from kfac_ferminet_alpha import layers_and_loss_tags as tags
from kfac_ferminet_alpha import utils


vmap_psd_inv_cholesky = jax.vmap(utils.psd_inv_cholesky, (0, None), 0)
vmap_matmul = jax.vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)


qmc1_tag = tags.LayerTag("qmc1_tag", 1, 1)


def register_qmc1(y, x, w, **kwargs):
  return qmc1_tag.bind(y, x, w, **kwargs)


qmc2_tag = tags.LayerTag("qmc2_tag", 1, 1)


def register_qmc2(y, x, w, **kwargs):
  return qmc2_tag.bind(y, x, w, **kwargs)


repeated_dense_tag = tags.LayerTag("repeated_dense_tag", 1, 1)


def register_repeated_dense(y, x, w, b):
  if b is None:
    return repeated_dense_tag.bind(y, x, w)
  return repeated_dense_tag.bind(y, x, w, b)


class QmcBlockedDense(blocks.TwoKroneckerFactored):
  """A factor that is the Kronecker product of two matrices."""

  def update_curvature_inverse_estimate(self, diagonal_weight, pmap_axis_name):
    self.inputs_factor.sync(pmap_axis_name)

    self.outputs_factor.sync(pmap_axis_name)
    vmap_pi_adjusted_inverse = jax.vmap(
        functools.partial(utils.pi_adjusted_inverse,
                          pmap_axis_name=pmap_axis_name),
        (0, 0, None), (0, 0)
    )
    self.inputs_factor_inverse, self.outputs_factor_inverse = (
        vmap_pi_adjusted_inverse(self.inputs_factor.value,
                                 self.outputs_factor.value,
                                 diagonal_weight / self.extra_scale))

  def multiply_matpower(self, vec, exp, diagonal_weight):
    w, = vec
    # kmjn
    v = w
    k, m, j, n = v.shape
    if exp == 1:
      inputs_factor = self.inputs_factor.value
      outputs_factor = self.outputs_factor.value
      scale = self.extra_scale
    elif exp == -1:
      inputs_factor = self.inputs_factor_inverse
      outputs_factor = self.outputs_factor_inverse
      scale = 1.0 / self.extra_scale
      diagonal_weight = 0.0
    else:
      raise NotImplementedError()
    # jk(mn)
    v = jnp.transpose(v, [2, 0, 1, 3]).reshape([j, k, m * n])
    v = vmap_matmul(inputs_factor, v)
    v = vmap_matmul(v, outputs_factor)
    # kmjn
    v = jnp.transpose(v.reshape([j, k, m, n]), [1, 2, 0, 3])
    v = v * scale + diagonal_weight * w
    return (v,)

  def update_curvature_matrix_estimate(
      self,
      info: blocks._BlockInfo,  # pylint: disable=protected-access
      batch_size: int,
      ema_old: Union[float, jnp.ndarray],
      ema_new: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    del pmap_axis_name
    (x,), (dy,) = info["inputs"], info["outputs_tangent"]
    assert batch_size == x.shape[0]
    normalizer = x.shape[0] * x.shape[1]
    # The forward computation is
    # einsum(x,w): bijk,bkmjn -> bijmn
    inputs_cov = jnp.einsum("bijk,bijl->jkl", x, x) / normalizer
    dy = jnp.reshape(dy, dy.shape[:-2] + (-1,))
    outputs_cov = jnp.einsum("bijk,bijl->jkl", dy, dy) / normalizer
    self.inputs_factor.update(inputs_cov, ema_old, ema_new)
    self.outputs_factor.update(outputs_cov, ema_old, ema_new)

  def init(self, rng):
    del rng
    k, m, j, n = self.params_shapes[0]
    return dict(
        inputs_factor=utils.WeightedMovingAverage.zero([j, k, k]),
        inputs_factor_inverse=jnp.zeros([j, k, k]),
        outputs_factor=utils.WeightedMovingAverage.zero([j, m * n, m * n]),
        outputs_factor_inverse=jnp.zeros([j, m * n, m * n]),
        extra_scale=jnp.asarray(m)
    )

  def input_size(self) -> int:
    raise NotImplementedError()

  def output_size(self) -> int:
    raise NotImplementedError()


class RepeatedDenseBlock(blocks.DenseTwoKroneckerFactored):
  """Dense block that is repeated."""

  def compute_extra_scale(self) -> Optional[jnp.ndarray]:
    (x_shape,) = self.inputs_shapes
    return utils.product(x_shape) // (x_shape[0] * x_shape[-1])

  def update_curvature_matrix_estimate(
      self,
      info: Mapping[str, blocks._Arrays],  # pylint: disable=protected-access
      batch_size: int,
      ema_old: Union[float, jnp.ndarray],
      ema_new: Union[float, jnp.ndarray],
      pmap_axis_name: str
  ) -> None:
    info = dict(**info)
    (x,), (dy,) = info["inputs"], info["outputs_tangent"]
    assert x.shape[0] == batch_size
    info["inputs"] = (x.reshape([-1, x.shape[-1]]),)
    info["outputs_tangent"] = (dy.reshape([-1, dy.shape[-1]]),)
    super().update_curvature_matrix_estimate(info, x.size // x.shape[-1],
                                             ema_old, ema_new, pmap_axis_name)


blocks.set_default_tag_to_block("qmc1_tag", QmcBlockedDense)
blocks.set_default_tag_to_block("repeated_dense_tag", RepeatedDenseBlock)
