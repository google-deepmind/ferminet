# Copyright 2024 DeepMind Technologies Limited.
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

"""Config to reproduce Fig. 1 from Pfau et al. (2024)."""

from ferminet import base_config
from ferminet.configs import atom
from ferminet.configs.excited import presets
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Returns config for running generic atoms with qmc."""
  cfg = base_config.default()
  cfg.system.atom = ''
  cfg.system.charge = 0
  cfg.system.delta_charge = 0.0
  cfg.system.states = 5
  cfg.system.spin_polarisation = ml_collections.FieldReference(
      None, field_type=int)
  cfg.pretrain.iterations = 10_000
  cfg.update_from_flattened_dict(presets.excited_states)
  with cfg.ignore_type():
    cfg.system.set_molecule = atom.adjust_nuclear_charge
  return cfg
