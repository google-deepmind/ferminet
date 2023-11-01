# Copyright 2023 DeepMind Technologies Limited.
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

"""Example excited states config for HCl with FermiNet and pseudopotentials."""

from ferminet import base_config
from ferminet.utils import system
import ml_collections
import pyscf


def finalize(cfg):
  """Sets the molecule, nuclear charge electrons for the atoms.

  Args:
    cfg: ml_collections.ConfigDict after all argument parsing.

  Returns:
    ml_collections.ConfictDict with the nuclear charge for the atom in
    cfg.system.molecule and cfg.system.charge appropriately set.
  """

  # Create a pyscf Mole object with pseudopotentials to be used for
  # pretraining and updating the config for consistency
  mol = pyscf.gto.Mole()
  mol.atom = [[atom.symbol, atom.coords] for atom in cfg.system.molecule]

  atoms = list(set([atom.symbol for atom in cfg.system.molecule]))
  pseudo_atoms = cfg.system.pp.symbols if cfg.system.use_pp else []
  mol.basis = {
      atom:
      cfg.system.pp.basis if atom in pseudo_atoms else 'cc-pvdz'
      for atom in atoms
  }
  mol.ecp = {
      atom: cfg.system.pp.type
      for atom in atoms if atom in pseudo_atoms
  }

  mol.charge = 0
  mol.spin = 0
  mol.unit = 'angstrom'
  mol.build()

  cfg.system.pyscf_mol = mol

  return cfg


def get_config():
  """Returns config for running generic atoms with qmc."""
  cfg = base_config.default()
  cfg.system.molecule = [
      system.Atom(symbol='H', coords=(0.0, 0.0, 0.0), units='angstrom'),
      system.Atom(symbol='Cl', coords=(0.0, 0.0, 1.2799799), units='angstrom'),
  ]
  cfg.system.electrons = (9, 9)  # Core electrons are removed automatically
  cfg.system.use_pp = True  # Enable pseudopotentials
  cfg.system.pp.symbols = ['Cl']  # Indicate which atoms to apply PP to
  cfg.system.charge = 0
  cfg.system.delta_charge = 0.0
  cfg.system.states = 3
  cfg.pretrain.iterations = 10_000
  cfg.optim.reset_if_nan = True
  cfg.system.spin_polarisation = ml_collections.FieldReference(
      None, field_type=int)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalize
    cfg.config_module = '.diatomic'
  return cfg
