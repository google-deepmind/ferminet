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
# limitations under the License.

"""Neural network building blocks."""

import itertools
from typing import Optional, Sequence

import jax
import jax.numpy as jnp


def array_partitions(sizes: Sequence[int]) -> Sequence[int]:
  """Returns the indices for splitting an array into separate partitions.

  Args:
    sizes: size of each of N partitions. The dimension of the array along
    the relevant axis is assumed to be sum(sizes).

  Returns:
    sequence of indices (length len(sizes)-1) at which an array should be split
    to give the desired partitions.
  """
  return list(itertools.accumulate(sizes))[:-1]


def linear_layer(x: jnp.ndarray,
                 w: jnp.ndarray,
                 b: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Evaluates a linear layer, x w + b.

  Args:
    x: inputs.
    w: weights.
    b: optional bias.

  Returns:
    x w + b if b is given, x w otherwise.
  """
  y = jnp.dot(x, w)
  return y + b if b is not None else y

vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)
