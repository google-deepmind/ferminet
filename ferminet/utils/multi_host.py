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

"""Generic utilities."""

from absl import logging
import jax
import jax.numpy as jnp


def check_synced(obj, name):
  """Checks whether the object is synced across local devices.

  Args:
    obj: PyTree with leaf nodes mapped over local devices.
    name: the name of the object (for logging only).

  Returns:
    True if object is in sync across all devices and False otherwise.
  """
  for i in range(1, jax.local_device_count()):
    norms = jax.tree.map(lambda x: jnp.linalg.norm(x[0] - x[i]), obj)  # pylint: disable=cell-var-from-loop
    total_norms = sum(jax.tree.leaves(norms))
    if total_norms != 0.0:
      logging.info(
          '%s object is not synced across device 0 and %d. The total norm'
          ' of the difference is %.5e. For specific detail inspect '
          'the individual differences norms:\n %s.',
          name, i, total_norms, str(norms)
      )
      return False
  logging.info('%s objects are synced.', name)
  return True
