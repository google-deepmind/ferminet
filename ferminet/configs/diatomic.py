# Copyright 2020 DeepMind Technologies Limited.
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

"""Diatomic molecule config for FermiNet."""

import re

from ferminet import base_config
from ferminet.utils import system


# Default bond lengths in angstrom for a few diatomics of interest.
# N2 bond length from the G3 dataset. Ref:
# 1. http://www.cse.anl.gov/OldCHMwebsiteContent/compmat/comptherm.htm
# 2. L. A. Curtiss, P. C. Redfern, K. Raghavachari, and J. A. Pople,
#    J. Chem. Phys, 109, 42 (1998).
BOND_LENGTHS = {
    'BeH': 1.348263,
    'CN': 1.134797,
    'ClF': 1.659091,
    'F2': 1.420604,
    'H2': 0.737164,
    'HCl': 1.2799799,
    'Li2': 2.77306,
    'LiH': 1.639999,
    'N2': 1.129978,
    'NH': 1.039428,
    'CO': 1.150338,
    'BH': 1.2324,  # Not in G3
    'PN': 1.491,  # Not in G3
    'AlH': 1.648,  # Not in G3
    'AlN': 1.786,  # Not in G3
}

# Default spin polarisation for a few diatomics of interest.
# Otherwise default to either singlet (doublet) for even (odd) numbers of
# electrons. Units: number of unpaired electrons.
SPIN_POLARISATION = {
    'B2': 2,
    'O2': 2,
    'NH': 2,
    'AlN': 2,
}


def molecule(cfg):
  """Creates molecule in cfg."""
  if cfg.system.molecule_name.endswith('2'):
    atom1 = atom2 = cfg.system.molecule_name.strip('2')
  else:
    atom1, atom2 = re.findall('[A-Z][a-z]*', cfg.system.molecule_name)
  if cfg.system.bond_length < 0:
    if cfg.system.molecule_name in BOND_LENGTHS:
      cfg.system.bond_length = BOND_LENGTHS[cfg.system.molecule_name]
    else:
      raise ValueError('bond length not set.')
  pos = (cfg.system.bond_length * cfg.system.bond_length_multiple) / 2
  atom_coords = ((-pos, 0., 0.), (pos, 0., 0.))
  cfg.system.molecule = [
      system.Atom(symbol=atom, coords=coord, units=cfg.system.units)
      for atom, coord in zip((atom1, atom2), atom_coords)
  ]

  if any(
      abs(atom.charge - round(atom.charge)) > 1e-6
      for atom in cfg.system.molecule):
    raise RuntimeError(
        'Cannot set the number of electrons for a fractional charge atom.')
  electrons = sum(int(round(atom.charge)) for atom in cfg.system.molecule)

  if not cfg.system.electrons:
    if cfg.system.molecule_name in SPIN_POLARISATION:
      spin = SPIN_POLARISATION[cfg.system.molecule_name]
    else:
      spin = electrons % 2
    nalpha = (electrons + spin) // 2
    cfg.system.electrons = (nalpha, electrons - nalpha)

  return cfg


def get_config():
  """Returns the config for running a diatomic molecule with qmc."""
  cfg = base_config.default()
  # Can specify homonuclear diatomics using X2 or heteronuclear diaomics using
  # XY.
  cfg.system.molecule_name = 'N2'

  cfg.system.bond_length = -1.0
  cfg.system.units = 'angstrom'
  cfg.system.bond_length_multiple = 1.0
  with cfg.ignore_type():
    cfg.system.set_molecule = molecule
    cfg.config_module = '.diatomic'

  return cfg
