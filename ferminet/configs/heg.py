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
# limitations under the License

"""Unpolarised 14 electron simple cubic homogeneous electron gas."""

from ferminet import base_config
from ferminet.utils import system

import numpy as np


def _sc_lattice_vecs(rs: float, nelec: int, ndim: int) -> np.ndarray:
  """Returns simple cubic lattice vectors with Wigner-Seitz radius rs."""
  if ndim == 2:
    area = np.pi * (rs**2) * nelec
    length = area**(1 / 2)
    return length * np.eye(2)
  elif ndim == 3:
    area = 4 * np.pi * (rs**3) * nelec / 3
    length = area**(1 / 3)
    return length * np.eye(3)
  else:
    raise NotImplementedError

rs = 1.0

def get_config():
  # Get default options.
  cfg = base_config.default()

  # SYSTEM
  cfg.system.electrons = (7, 7)
  # A ghost atom at the origin defines one-electron coordinate system.
  # Element 'X' is a dummy nucleus with zero charge
  cfg.system.molecule = [system.Atom('X', (0, 0, 0))]

  cfg.system.lattice = _sc_lattice_vecs(rs, sum(cfg.system.electrons), cfg.system.ndim)
  cfg.network.make_feature_layer_kwargs['include_r_ae'] = False

  # Pretraining is not currently implemented for systems in PBC
  cfg.pretrain.method = None

  return cfg
