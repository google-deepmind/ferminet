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

"""Generic single-atom configuration for FermiNet."""

from ferminet import base_config
from ferminet.utils import elements
from ferminet.utils import system
import ml_collections


def adjust_nuclear_charge(cfg):
  """Sets the molecule, nuclear charge electrons for the atom.

  Note: function name predates this logic but is kept for compatibility with
  xm_expt.py.

  Args:
    cfg: ml_collections.ConfigDict after all argument parsing.

  Returns:
    ml_collections.ConfictDict with the nuclear charge for the atom in
    cfg.system.molecule and cfg.system.charge appropriately set.
  """
  if cfg.system.molecule:
    atom = cfg.system.molecule[0]
  else:
    atom = system.Atom(symbol=cfg.system.atom, coords=(0, 0, 0))

  if abs(cfg.system.delta_charge) > 1.e-8:
    nuclear_charge = atom.charge + cfg.system.delta_charge
    cfg.system.molecule = [
        system.Atom(atom.symbol, atom.coords, nuclear_charge)
    ]
  else:
    cfg.system.molecule = [atom]

  if not cfg.system.electrons:
    atomic_number = elements.SYMBOLS[atom.symbol].atomic_number
    if 'charge' in cfg.system:
      atomic_number -= cfg.system.charge
    if ('spin_polarisation' in cfg.system
        and cfg.system.spin_polarisation is not None):
      spin_polarisation = cfg.system.spin_polarisation
    else:
      spin_polarisation = elements.ATOMIC_NUMS[atomic_number].spin_config
    nalpha = (atomic_number + spin_polarisation) // 2
    cfg.system.electrons = (nalpha, atomic_number - nalpha)

  return cfg


def get_config():
  """Returns config for running generic atoms with qmc."""
  cfg = base_config.default()
  cfg.system.atom = ''
  cfg.system.charge = 0
  cfg.system.delta_charge = 0.0
  cfg.system.spin_polarisation = ml_collections.FieldReference(
      None, field_type=int)
  with cfg.ignore_type():
    cfg.system.set_molecule = adjust_nuclear_charge
    cfg.config_module = '.atom'
  return cfg
