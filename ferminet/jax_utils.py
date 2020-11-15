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

"""Utilities for working with JAX."""

import jax

broadcast = jax.pmap(lambda x: x)

p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))


def replicate(pytree):
  n = jax.local_device_count()
  stacked_pytree = jax.tree_map(lambda x: jax.lax.broadcast(x, (n,)), pytree)
  return broadcast(stacked_pytree)
