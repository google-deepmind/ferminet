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

"""Config to reproduce Fig. 5 from Pfau et al. (2024)."""

from ferminet import base_config
from ferminet.configs.excited import presets
from ferminet.utils import system
import ml_collections


# Geometries are taken from Loos, Boggio-Pasqua, Scemama, Caffarel and
# Jacquemin, JCTC (2019). The units listed in the supplemental material are
# inconsistent: nitrosomethane and glyoxal are in Bohr, the rest are Angstrom.
_SYSTEMS = {
    'nitrosomethane': """C -1.78426612 0.00000000 -1.07224050
N -0.00541753 0.00000000 1.08060391
O 2.18814985 0.00000000 0.43452135
H -0.77343975 0.00000000 -2.86415606
H -2.97471478 1.66801808 -0.86424584
H -2.97471478 -1.66801808 -0.86424584""".split('\n'),
    'butadiene': """C 1.740343 0.616556 0.00000000
C -1.740343 -0.616556 0.00000000
C 0.397343 0.616556 0.00000000
C -0.397343 -0.616556 0.00000000
H 0.126346 -1.577069 0.00000000
H -0.126346 1.577069 0.00000000
H 2.279054 1.568725 0.00000000
H -2.279054 -1.568725 0.00000000
H 2.279054 -0.335614 0.00000000
H -2.279054 0.335614 0.00000000""".split('\n'),
    'glyoxal': """C 1.21360282 0.75840215 0.00000000
C -1.21360282 -0.75840215 0.00000000
O 3.25581408 -0.26453186 0.00000000
O -3.25581408 0.26453186 0.00000000
H 0.96135276 2.81883243 0.00000000
H -0.96135276 -2.81883243 0.00000000""".split('\n'),
    'tetrazine': """C  0.00000000  0.00000000   1.26054332
C  0.00000000  0.00000000  -1.26054332
N  0.00000000  1.19421138   0.66133002
N  0.00000000 -1.19421138   0.66133002
N  0.00000000  1.19421138  -0.66133002
N  0.00000000 -1.19421138  -0.66133002
H  0.00000000  0.00000000   2.33817427
H  0.00000000  0.00000000  -2.33817427""".split('\n'),
    'cyclopentadienone': """C  0.00000000  0.00000000   0.76853878
C  0.00000000  1.19974276  -0.13448057
C  0.00000000 -1.19974276  -0.13448057
C  0.00000000  0.74909075  -1.39624830
C  0.00000000 -0.74909075  -1.39624830
O  0.00000000  0.00000000   1.98144505
H  0.00000000  2.21416694   0.22305399
H  0.00000000 -2.21416694   0.22305399
H  0.00000000  1.34284493  -2.29584273
H  0.00000000 -1.34284493  -2.29584273""".split('\n')
}


def finalise(
    experiment_config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Returns the experiment config with the molecule set."""
  system_name = experiment_config.system.molecule_name
  geom = _SYSTEMS[system_name]
  units = 'bohr' if system_name in ['gloyxal', 'nitrosomethane'] else 'angstrom'

  molecule = []
  for atom in geom:
    element, x, y, z = atom.split()
    coords = [float(xx) for xx in (x, y, z)]
    molecule.append(system.Atom(symbol=element,
                                coords=coords,
                                units=units))

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
  cfg.system.molecule_name = 'nitrosomethane'
  cfg.system.states = 6
  cfg.system.spin_polarisation = ml_collections.FieldReference(
      None, field_type=int)
  cfg.pretrain.iterations = 50_000
  cfg.mcmc.blocks = 2
  cfg.update_from_flattened_dict(presets.excited_states)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalise
  return cfg
