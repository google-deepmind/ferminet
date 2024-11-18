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
import dataclasses
import functools
from typing import Sequence, Set, Tuple
import jax
import jax.numpy as jnp
import kfac_jax
import numpy as np


Array = kfac_jax.utils.Array
Scalar = kfac_jax.utils.Scalar
Numeric = kfac_jax.utils.Numeric

vmap_psd_inv = jax.vmap(kfac_jax.utils.psd_inv, (0, None), 0)
vmap_matmul = jax.vmap(jnp.matmul, in_axes=(0, 0), out_axes=0)


def register_repeated_dense(y, x, w, b, **kwargs):
  return kfac_jax.register_dense(y, x, w, b, variant="repeated_dense", **kwargs)


def register_qmc(y, x, w, **kwargs):
  return kfac_jax.register_dense(y, x, w, variant="qmc", **kwargs)


_dense = functools.partial(
    kfac_jax.tag_graph_matcher._dense,  # pylint: disable=protected-access
    axes=1,
    with_reshape=False,
)


_repeated_dense_parameter_extractor = functools.partial(
    kfac_jax.tag_graph_matcher._dense_parameter_extractor,  # pylint: disable=protected-access
    variant="repeated_dense",
)


class RepeatedDenseBlock(kfac_jax.DenseTwoKroneckerFactored):
  """Dense block that is repeatedly applied to multiple inputs (e.g. vmap)."""

  def fixed_scale(self) -> Numeric:
    (x_shape,) = self.inputs_shapes
    return float(kfac_jax.utils.product(x_shape) // (x_shape[0] * x_shape[-1]))

  def update_curvature_matrix_estimate(
      self,
      state: kfac_jax.KroneckerFactored.State,
      estimation_data: kfac_jax.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: int,
  ) -> kfac_jax.KroneckerFactored.State:
    [x] = estimation_data.primals.inputs
    [dy] = estimation_data.tangents.outputs
    assert x.shape[0] == batch_size

    estimation_data = dataclasses.replace(
        estimation_data,
        primals=dataclasses.replace(
            estimation_data.primals,
            inputs=(x.reshape([-1, x.shape[-1]]),),
        ),
        tangents=dataclasses.replace(
            estimation_data.tangents,
            outputs=(dy.reshape([-1, dy.shape[-1]]),),
        ),
    )

    batch_size = x.size // x.shape[-1]
    return super().update_curvature_matrix_estimate(
        state=state,
        estimation_data=estimation_data,
        ema_old=ema_old,
        ema_new=ema_new,
        identity_weight=identity_weight,
        batch_size=batch_size,
    )


class QmcBlockedDense(kfac_jax.KroneckerFactored):
  """A factor that is the Kronecker product of two matrices."""

  def input_size(self) -> int:
    raise NotImplementedError()

  def output_size(self) -> int:
    raise NotImplementedError()

  def fixed_scale(self) -> Numeric:
    return float(self.parameters_shapes[0][1])

  def update_curvature_matrix_estimate(
      self,
      state: kfac_jax.KroneckerFactored.State,
      estimation_data: kfac_jax.LayerVjpData[Array],
      ema_old: Numeric,
      ema_new: Numeric,
      identity_weight: Numeric,
      batch_size: int,
  ) -> kfac_jax.KroneckerFactored.State:
    del identity_weight

    [x] = estimation_data.primals.inputs
    [dy] = estimation_data.tangents.outputs
    assert batch_size == x.shape[0]
    normalizer = x.shape[0] * x.shape[1]
    # The forward computation is
    # einsum(x,w): bijk,bkmjn -> bijmn
    inputs_cov = jnp.einsum("bijk,bijl->jkl", x, x) / normalizer
    dy = jnp.reshape(dy, dy.shape[:-2] + (-1,))
    outputs_cov = jnp.einsum("bijk,bijl->jkl", dy, dy) / normalizer

    state.inputs_factor.update(inputs_cov, ema_old, ema_new)
    state.outputs_factor.update(outputs_cov, ema_old, ema_new)

    return state

  def _init(
      self,
      rng: kfac_jax.utils.PRNGKey,
      exact_powers_to_cache: Set[Scalar],
      approx_powers_to_cache: Set[Scalar],
      cache_eigenvalues: bool,
  ) -> kfac_jax.KroneckerFactored.State:
    del rng, cache_eigenvalues
    k, m, j, n = self.parameters_shapes[0]
    cache = dict()
    if exact_powers_to_cache:
      raise NotImplementedError(
          "Caching of exact powers is not yet implemented for QmcBlockedDense.")
    for power in approx_powers_to_cache:
      if power != -1:
        raise NotImplementedError(f"Approximations for power {power} is not "
                                  f"yet implemented.")
      cache[str(power)] = dict(
          inputs_factor=jnp.zeros([j, k, k]),
          outputs_factor=jnp.zeros([j, m * n, m * n]),
      )
    return kfac_jax.KroneckerFactored.State(
        cache=cache,
        inputs_factor=
        kfac_jax.utils.WeightedMovingAverage.zeros_array((j, k, k)),
        outputs_factor=kfac_jax.utils.WeightedMovingAverage.zeros_array(
            (j, m * n, m * n)),
    )

  def _update_cache(
      self,
      state: kfac_jax.KroneckerFactored.State,
      identity_weight: kfac_jax.utils.Numeric,
      exact_powers: set[kfac_jax.utils.Scalar],
      approx_powers: set[kfac_jax.utils.Scalar],
      eigenvalues: bool,
  ) -> kfac_jax.KroneckerFactored.State:
    del eigenvalues

    if exact_powers:
      raise NotImplementedError(
          "Caching of exact powers is not yet implemented for QmcBlockedDense.")
    for power in approx_powers:
      if power != -1:
        raise NotImplementedError(f"Approximations for power {power} is not "
                                  f"yet implemented.")
      cache = state.cache[str(power)]
      pi_adjusted_inverse = jax.vmap(
          kfac_jax.utils.pi_adjusted_kronecker_inverse,
          (0, None), (0, 0)
      )
      cache["inputs_factor"], cache["outputs_factor"] = pi_adjusted_inverse(
          state.inputs_factor.value,
          state.outputs_factor.value,
          damping=identity_weight,
      )
    return state

  def multiply_matpower(
      self,
      state: kfac_jax.KroneckerFactored.State,
      vector: Sequence[Array],
      identity_weight: Numeric,
      power: Scalar,
      exact_power: bool,
      use_cached: bool,
  ) -> Tuple[Array, ...]:
    w, = vector
    # kmjn
    v = w
    k, m, j, n = v.shape
    if power == 1:
      # jk(mn)
      v = jnp.transpose(v, [2, 0, 1, 3]).reshape([j, k, m * n])
      v = vmap_matmul(state.inputs_factor.value, v)
      v = vmap_matmul(v, state.outputs_factor.value)
      # kmjn
      v = jnp.transpose(v.reshape([j, k, m, n]), [1, 2, 0, 3])
      v = v + identity_weight * w
    elif exact_power:
      raise NotImplementedError(
          "Exact powers is not yet implemented for QmcBlockedDense.")
    else:
      if not use_cached:
        raise NotImplementedError(
            "Caching of exact powers is not yet implemented for "
            "QmcBlockedDense.")
      else:
        # jk(mn)
        v = jnp.transpose(v, [2, 0, 1, 3]).reshape([j, k, m * n])
        v = vmap_matmul(state.cache[str(power)]["inputs_factor"], v)
        v = vmap_matmul(v, state.cache[str(power)]["outputs_factor"])
        # kmjn
        v = jnp.transpose(v.reshape([j, k, m, n]), [1, 2, 0, 3])
    # kmjn
    return (v,)


# repeating a dense layer once
_repeated_dense1 = jax.vmap(_dense, in_axes=[0, [None, None]])
_repeated_dense2 = jax.vmap(_repeated_dense1, in_axes=[0, [None, None]])
_repeated_dense1_no_b = jax.vmap(_dense, in_axes=[0, [None]])
_repeated_dense2_no_b = jax.vmap(_repeated_dense1_no_b, in_axes=[0, [None]])

# Computation for repeated dense layer
repeated_dense1_with_bias_pattern = kfac_jax.tag_graph_matcher.GraphPattern(
    name="repeated_dense1_with_bias",
    tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
    compute_func=_repeated_dense1,
    parameters_extractor_func=_repeated_dense_parameter_extractor,
    example_args=[np.zeros([9, 11, 13]), [np.zeros([13, 7]), np.zeros([7])]],
)

repeated_dense1_no_bias_pattern = kfac_jax.tag_graph_matcher.GraphPattern(
    name="repeated_dense1_no_bias",
    tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
    compute_func=_repeated_dense1_no_b,
    parameters_extractor_func=_repeated_dense_parameter_extractor,
    example_args=[np.zeros([9, 11, 13]), [np.zeros([13, 7])]],
)

repeated_dense2_with_bias_pattern = kfac_jax.tag_graph_matcher.GraphPattern(
    name="repeated_dense2_with_bias",
    tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
    compute_func=_repeated_dense2,
    parameters_extractor_func=_repeated_dense_parameter_extractor,
    example_args=[np.zeros([8, 9, 11, 13]), [np.zeros([13, 7]), np.zeros([7])]],
)

repeated_dense2_no_bias_pattern = kfac_jax.tag_graph_matcher.GraphPattern(
    name="repeated_dense2_no_bias",
    tag_primitive=kfac_jax.layers_and_loss_tags.layer_tag,
    compute_func=_repeated_dense2_no_b,
    parameters_extractor_func=_repeated_dense_parameter_extractor,
    example_args=[np.zeros([8, 9, 11, 13]), [np.zeros([13, 7])]],
)

GRAPH_PATTERNS = (
    repeated_dense1_with_bias_pattern,
    repeated_dense2_with_bias_pattern,
    repeated_dense1_no_bias_pattern,
    repeated_dense2_no_bias_pattern,
) + kfac_jax.tag_graph_matcher.DEFAULT_GRAPH_PATTERNS


kfac_jax.set_default_tag_to_block_ctor(
    "repeated_dense", RepeatedDenseBlock
)
kfac_jax.set_default_tag_to_block_ctor("qmc", QmcBlockedDense)
