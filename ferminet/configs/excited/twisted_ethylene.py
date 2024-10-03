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

"""Config to reproduce Fig. 4 from Pfau et al. (2024)."""

from ferminet import base_config
from ferminet.configs.excited import presets
from ferminet.utils import system
import ml_collections
import numpy as np


# Geometries from Entwistle, Schätzle, Erdman, Hermann and Noé (2023), taken
# from Barbatti, Paier and Lischka (2004). All geometries are in Angstroms.
_SYSTEMS = {
    'planar': ['C -0.675 0.0 0.0',
               'C 0.675 0.0 0.0',
               'H -1.2429 0.0 -0.93037',
               'H -1.2429 0.0 0.93037',
               'H 1.2429 0.0 -0.93037',
               'H 1.2429 0.0 0.93037'],
    'twisted': ['C -0.6885 0.0 0.0',
                'C 0.6885 0.0 0.0',
                'H -1.307207 0.0 -0.915547',
                'H -1.307207 0.0 0.915547',
                'H 1.307207 -0.915547 0.0',
                'H 1.307207 0.915547 0.0'],
}


def finalise(
    experiment_config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Returns the experiment config with the molecule set."""
  geom = _SYSTEMS[experiment_config.system.molecule_name]

  molecule = []
  for i, atom in enumerate(geom):
    element, x, y, z = atom.split()
    coords = np.array([float(xx) * 1.88973 for xx in (x, y, z)])  # ang to bohr
    if i == 4 or i == 5:
      if experiment_config.system.twist.tau != 0:
         # rotate hydrogens around x axis
        tau = experiment_config.system.twist.tau * np.pi / 180.0  # deg to rad
        rot = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(tau), -np.sin(tau)],
                        [0.0, np.sin(tau), np.cos(tau)]])
        coords = rot @ coords
      if experiment_config.system.twist.phi != 0:
         # rotate hydrogens around y axis
        phi = experiment_config.system.twist.phi * np.pi / 180.0  # deg to rad
        rot = np.array([[np.cos(phi), 0.0, -np.sin(phi)],
                        [0.0, 1.0, 0.0],
                        [np.sin(phi), 0.0, np.cos(phi)]])
        # position of carbon atom
        coord0 = np.array([float(xx) for xx in geom[1].split()[1:]]) * 1.88973
        coords = (rot @ (coords - coord0)) + coord0
    molecule.append(system.Atom(symbol=element,
                                coords=list(coords),
                                units='bohr'))

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
  cfg.system.molecule_name = 'planar'
  cfg.system.twist = {
      'tau': 0.0,  # torsion angle, in degrees
      'phi': 0.0,  # pyramidalization angle, in degrees
  }
  cfg.system.states = 3  # Note that for equilibrium only, we computed 5 states.
  cfg.system.spin_polarisation = ml_collections.FieldReference(
      None, field_type=int)
  cfg.system.units = 'bohr'
  cfg.system.electrons = (8, 8)
  cfg.pretrain.iterations = 10_000
  cfg.optim.iterations = 100_000
  cfg.update_from_flattened_dict(presets.psiformer)
  cfg.update_from_flattened_dict(presets.excited_states)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalise
  return cfg
