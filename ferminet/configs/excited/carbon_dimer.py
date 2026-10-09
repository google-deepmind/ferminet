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

"""Config to reproduce Fig. 3 from Pfau et al. (2024)."""

from ferminet import base_config
from ferminet.configs.excited import presets
from ferminet.utils import system
import ml_collections


def finalise(
    experiment_config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Returns the experiment config with the molecule set."""
  # Equilibrium bond length is 1.244 Angstrom
  bond_length = experiment_config.system.equilibrium_multiple * 1.244 * 1.88973
  experiment_config.system.molecule = [
      system.Atom('C', coords=(0, 0, bond_length / 2)),
      system.Atom('C', coords=(0, 0, -bond_length / 2))]
  return experiment_config


def get_config() -> ml_collections.ConfigDict:
  """Returns config for running generic atoms with qmc."""
  cfg = base_config.default()
  cfg.system.charge = 0
  cfg.system.delta_charge = 0.0
  cfg.system.molecule_name = 'C2'
  cfg.system.states = 8
  cfg.system.spin_polarisation = ml_collections.FieldReference(
      None, field_type=int)
  cfg.system.units = 'bohr'
  cfg.system.electrons = (6, 6)
  cfg.pretrain.iterations = 100_000
  cfg.optim.iterations = 100_000
  cfg.update_from_flattened_dict(presets.psiformer)
  cfg.update_from_flattened_dict(presets.excited_states)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalise
  return cfg
