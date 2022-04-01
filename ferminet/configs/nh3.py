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

"""Ammonia example config."""

from ferminet import base_config
from ferminet.utils import system
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Returns config for running NH3 with FermiNet."""
  cfg = base_config.default()
  # geometry in bohr.
  cfg.system.molecule = [
      system.Atom(symbol='N', coords=(0.0, 0.0, 0.22013)),
      system.Atom(symbol='H', coords=(0.0, 1.77583, -0.51364)),
      system.Atom(symbol='H', coords=(1.53791, -0.88791, -0.51364)),
      system.Atom(symbol='H', coords=(-1.53791, -0.88791, -0.51364)),
  ]
  cfg.system.electrons = (5, 5)
  return cfg
