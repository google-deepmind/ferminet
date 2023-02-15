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

"""Constants for FermiNet."""

import functools
import jax
import kfac_jax


# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)

# Shortcut for kfac utils
psum = functools.partial(kfac_jax.utils.psum_if_pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(
    kfac_jax.utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)
all_gather = functools.partial(kfac_jax.utils.wrap_if_pmap(jax.lax.all_gather),
                               axis_name=PMAP_AXIS_NAME)
