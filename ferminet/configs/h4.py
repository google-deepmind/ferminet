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

"""Four hydrogen atoms on a circle."""

import itertools
from ferminet import base_config
from ferminet.utils import system
import ml_collections
import numpy as np


def _set_geometry(cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Returns the config with the H4 molecule set."""
  t = np.radians(cfg.system.angle / 2)
  x = cfg.system.radius * np.cos(t)
  y = cfg.system.radius * np.sin(t)
  quadrants = itertools.product((1, -1), (1, -1))
  cfg.system.molecule = [
      system.Atom(
          symbol='H', coords=(i * x, j * y, 0.0), units=cfg.system.units)
      for i, j in quadrants
  ]

  return cfg


def get_config():
  """Returns config for running H4 with FermiNet."""
  cfg = base_config.default()
  cfg.system.update({
      'angle': 90,
      'radius': 1.738,
      'units': 'angstrom',
      'electrons': (2, 2),
  })
  with cfg.ignore_type():
    cfg.system.set_molecule = _set_geometry
    cfg.config_module = '.h4'
  return cfg
