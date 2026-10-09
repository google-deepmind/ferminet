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

"""1D hydrogen chain."""

from ferminet import base_config
from ferminet.utils import system
import ml_collections


def _set_geometry(cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Returns the config with the Hn molecule set."""
  start = -(cfg.system.bond_length * (cfg.system.natoms - 1)) / 2
  atom_position = lambda i: (start + i * cfg.system.bond_length, 0, 0)
  cfg.system.molecule = [
      system.Atom(symbol='H', coords=atom_position(i), units=cfg.system.units)
      for i in range(cfg.system.natoms)
  ]
  nalpha = cfg.system.natoms // 2
  cfg.system.electrons = (nalpha, cfg.system.natoms - nalpha)
  return cfg


def get_config():
  """Returns config for running Hn with FermiNet."""
  cfg = base_config.default()
  cfg.system.update({
      'bond_length': 1.4,
      'natoms': 2,
  })
  with cfg.ignore_type():
    cfg.system.set_molecule = _set_geometry
    cfg.config_module = '.h4'
  return cfg
