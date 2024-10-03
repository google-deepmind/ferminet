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

"""Config to reproduce Figs. 2 and S3 from Pfau et al. (2024)."""


from ferminet import base_config
from ferminet.configs.excited import presets
from ferminet.utils import system
import ml_collections
import pyscf


# Geometries from Chrayteh, Blondel, Loos and Jacquemin, JCTC (2021)
# All geometries in atomic units (Bohr)
_SYSTEMS = {
    'BH': ['B 0.00000000 0.00000000 0.00000000',
           'H 0.00000000 0.00000000 2.31089693'],
    'HCl': ['H 0.00000000 0.00000000 2.38483140',
            'Cl 0.00000000 0.00000000 -0.02489783'],
    'H2O': ['O 0.00000000 0.00000000 -0.13209669',
            'H 0.00000000 1.43152878 0.97970006',
            'H 0.00000000 -1.43152878 0.97970006'],
    'H2S': ['S 0.00000000 0.00000000 -0.50365086',
            'H 0.00000000 1.81828105 1.25212288',
            'H 0.00000000 -1.81828105 1.25212288'],
    'BF': ['B 0.00000000 0.00000000 0.00000000',
           'F 0.00000000 0.00000000 2.39729626'],
    'CO': ['C 0.00000000 0.00000000 -1.24942055',
           'O 0.00000000 0.00000000 0.89266692'],
    'N2': ['N 0.00000000 0.00000000 1.04008632',
           'N 0.00000000 0.00000000 -1.04008632'],
    'C2H4': ['C 0.00000000 1.26026583 0.00000000',
             'C 0.00000000 -1.26026583 0.00000000',
             'H 0.00000000 2.32345976 1.74287672',
             'H 0.00000000 -2.32345976 1.74287672',
             'H 0.00000000 2.32345976 -1.74287672',
             'H 0.00000000 -2.32345976 -1.74287672'],
    'CH2O': ['C 0.00000000 0.00000000 1.13947666',
             'O 0.00000000 0.00000000 -1.14402883',
             'H 0.00000000 1.76627623 2.23398653',
             'H 0.00000000 -1.76627623 2.23398653'],
    'CH2S': ['C 0.00000000 0.00000000 2.08677304',
             'S 0.00000000 0.00000000 -0.97251194',
             'H 0.00000000 1.73657773 3.17013507',
             'H 0.00000000 -1.73657773 3.17013507'],
    'HNO': ['O 0.21099695 0.00000000 2.15462460',
            'N -0.44776863 0.00000000 -0.03589263',
            'H 1.18163475 0.00000000 -1.17386890'],
    'HCF': ['C -0.13561085 0.00000000 1.20394474',
            'F 1.85493976 0.00000000 -0.27610752',
            'H -1.71932891 0.00000000 -0.18206846'],
    'H2CSi': ['C 0.00000000 0.00000000 -2.09539928',
              'Si 0.00000000 0.00000000 1.14992930',
              'H 0.00000000 1.70929524 -3.22894481',
              'H 0.00000000 -1.70929524 -3.22894481'],
}


def finalise(
    experiment_config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Returns the experiment config with the molecule commpletely set."""
  geom = _SYSTEMS[experiment_config.system.molecule_name]

  molecule = []
  for atom in geom:
    element, x, y, z = atom.split()
    coords = [float(xx) for xx in (x, y, z)]
    molecule.append(system.Atom(symbol=element, coords=coords, units='bohr'))

  if not experiment_config.system.electrons:  # Don't override if already set
    nelectrons = int(sum(atom.charge for atom in molecule))
    na = nelectrons // 2
    experiment_config.system.electrons = (na, nelectrons - na)
  experiment_config.system.molecule = molecule

  pseudo_atoms = ['S', 'Si', 'Cl']  # use ECP for second-row atoms

  mol = pyscf.gto.Mole()
  mol.atom = [[atom.symbol, atom.coords] for atom in molecule]

  atoms = list(set([atom.symbol for atom in molecule]))
  mol.basis = {
      atom:
      experiment_config.system.ecp_basis if atom in pseudo_atoms else 'cc-pvdz'
      for atom in atoms
  }
  mol.ecp = {
      atom: experiment_config.system.ecp
      for atom in atoms if atom in pseudo_atoms
  }

  mol.charge = 0
  mol.spin = 0
  mol.unit = 'bohr'
  mol.build()

  experiment_config.system.pyscf_mol = mol
  experiment_config.system.make_local_energy_kwargs = {
      'ecp_symbols': list(mol.ecp.keys()),
      'ecp_type': experiment_config.system.ecp,
  }

  return experiment_config


def get_config() -> ml_collections.ConfigDict:
  """Returns the config for running FermiNet on a molecule from the G3 set."""
  cfg = base_config.default()

  cfg.system.molecule_name = ''
  cfg.system.ecp = 'ccecp'
  cfg.system.ecp_basis = 'ccecp-cc-pVDZ'
  cfg.system.states = 5
  cfg.pretrain.iterations = 10_000
  cfg.update_from_flattened_dict(presets.excited_states)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalise

  return cfg
