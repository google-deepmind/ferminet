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

"""Config to reproduce Fig. 6 from Pfau et al. (2024)."""

from ferminet import base_config
from ferminet.configs.excited import presets
from ferminet.utils import system
import ml_collections


_GEOM = """C    0.00000000            2.63144965            0.00000000
C   -2.27890225            1.31572483            0.00000000
C   -2.27890225           -1.31572483            0.00000000
C    0.00000000           -2.63144965            0.00000000
C    2.27890225           -1.31572483            0.00000000
C    2.27890225            1.31572483            0.00000000
H   -4.04725813            2.33668557            0.00000000
H   -4.04725813           -2.33668557            0.00000000
H   -0.00000000           -4.67337115            0.00000000
H    4.04725813           -2.33668557            0.00000000
H    4.04725813            2.33668557            0.00000000
H    0.00000000            4.67337115            0.00000000""".split('\n')


def finalise(
    experiment_config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Returns the experiment config with the molecule commpletely set."""
  molecule = []
  for atom in _GEOM:
    element, x, y, z = atom.split()
    coords = [float(xx) for xx in (x, y, z)]
    molecule.append(system.Atom(symbol=element, coords=coords, units='bohr'))

  if not experiment_config.system.electrons:  # Don't override if already set
    nelectrons = int(sum(atom.charge for atom in molecule))
    na = nelectrons // 2
    experiment_config.system.electrons = (na, nelectrons - na)
  experiment_config.system.molecule = molecule
  return experiment_config


def get_config() -> ml_collections.ConfigDict:
  """Returns config for running generic atoms with qmc."""
  cfg = base_config.default()
  cfg.system.charge = 0
  cfg.system.delta_charge = 0.0
  cfg.system.states = 6
  cfg.system.spin_polarisation = ml_collections.FieldReference(
      None, field_type=int)
  # While this value was used in the paper, it can be lowered.
  cfg.pretrain.iterations = 100_000
  cfg.mcmc.blocks = 4
  # While this envelope was used in the paper, it can be replaced with the
  # default 'isotropic' envelope without any noticeable change in the results.
  cfg.network.envelope_type = 'bottleneck'
  cfg.network.num_envelopes = 32
  cfg.update_from_flattened_dict(presets.excited_states)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalise

  return cfg
