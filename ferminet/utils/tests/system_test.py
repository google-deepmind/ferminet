# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ferminet.utils.system."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet.utils import system
from ferminet.utils import units
import numpy as np
import pyscf


class SystemAtomCoordsTest(absltest.TestCase):

  def test_atom_coords(self):
    xs = np.random.uniform(size=3)
    atom = system.Atom(symbol='H', coords=xs, units='angstrom')
    np.testing.assert_allclose(atom.coords / xs, [units.BOHR_ANGSTROM]*3)
    np.testing.assert_allclose(atom.coords_angstrom, xs)

  def test_atom_units(self):
    system.Atom(symbol='H', coords=[1, 2, 3], units='bohr')
    system.Atom(symbol='H', coords=[1, 2, 3], units='angstrom')
    with self.assertRaises(ValueError):
      system.Atom(symbol='H', coords=[1, 2, 3], units='dummy')


class SystemCreationsTest(parameterized.TestCase):

  @parameterized.parameters(
      {'symbol': 'He', 'charge': 0},
      {'symbol': 'C', 'charge': 0},
      {'symbol': 'Ne', 'charge': 0},
      {'symbol': 'Ne', 'charge': 1},
      {'symbol': 'Ne', 'charge': -1},
  )
  def test_create_atom(self, symbol, charge):
    mol, spins = system.atom(symbol, charge=charge)
    self.assertLen(mol, 1)
    self.assertEqual(mol[0].symbol, symbol)
    self.assertEqual(sum(spins), mol[0].atomic_number - charge)
    np.testing.assert_allclose(np.asarray(mol[0].coords), np.zeros(3))

  @parameterized.parameters(
      {'symbol': 'LiH'},
      {'symbol': 'Li2'},
      {'symbol': 'N2'},
      {'symbol': 'CO'},
      {'symbol': 'CH4'},
      {'symbol': 'NH3'},
      {'symbol': 'C2H4'},
      {'symbol': 'C4H6'},
  )
  def test_create_molecule(self, symbol):
    _, _ = system.molecule(symbol)

  @parameterized.parameters(
      {'n': 10, 'r': 1.0},
      {'n': 11, 'r': 1.0},
      {'n': 20, 'r': 2.0},
  )
  def test_create_hydrogen_chain(self, n, r):
    mol, spins = system.hn(n, r)
    self.assertLen(mol, n)
    for atom in mol:
      self.assertAlmostEqual(atom.coords[0], 0)
      self.assertAlmostEqual(atom.coords[1], 0)
    for atom1, atom2 in zip(mol[:-1], mol[1:]):
      self.assertAlmostEqual(atom2.coords[2] - atom1.coords[2], r)
    self.assertEqual(spins, (n - n // 2, n // 2))

  @parameterized.parameters(
      {'r': 1.0, 'angle': np.pi/4.0},
      {'r': 1.0, 'angle': np.pi/6.0},
      {'r': 2.0, 'angle': np.pi/4.0},
  )
  def test_create_hydrogen_circle(self, r, angle):
    mol, spins = system.h4_circle(r, angle)
    self.assertEqual(spins, (2, 2))
    self.assertLen(mol, 4)
    for atom in mol:
      self.assertAlmostEqual(atom.coords[2], 0)
      theta = np.abs(np.arctan(atom.coords[1] / atom.coords[0]))
      self.assertAlmostEqual(theta, angle)


class PyscfConversionTest(parameterized.TestCase):

  @parameterized.parameters([
      {
          'mol_string': 'H 0 0 -1; H 0 0 1'
      },
      {
          'mol_string': 'O 0 0 0; H  0 1 0; H 0 0 1'
      },
      {
          'mol_string': 'H 0 0 0; Cl 0 0 1.1'
      },
  ])
  def test_conversion_pyscf(self, mol_string):
    mol = pyscf.gto.Mole()
    mol.build(
        atom=mol_string,
        basis='sto-3g', unit='bohr')
    cfg = system.pyscf_mol_to_internal_representation(mol)
    # Assert that the alpha and beta electrons are the same
    self.assertEqual(mol.nelec, cfg.system.electrons)
    # Assert that the basis are the same
    self.assertEqual(mol.basis, cfg.pretrain.basis)
    # Assert that atom symbols are the same
    self.assertEqual([mol.atom_symbol(i) for i in range(mol.natm)],
                     [atom.symbol for atom in cfg.system.molecule])
    # Assert that atom coordinates are the same
    pyscf_coords = [mol.atom_coords()[i] for i in range(mol.natm)]
    internal_coords = [np.array(atom.coords) for atom in cfg.system.molecule]
    np.testing.assert_allclose(pyscf_coords, internal_coords)

  def test_conversion_pyscf_ang(self):
    mol = pyscf.gto.Mole()
    mol.build(
        atom='H 0 0 -1; H 0 0 1',
        basis='sto-3g', unit='ang')
    cfg = system.pyscf_mol_to_internal_representation(mol)
    # Assert that the coordinates are now in bohr internally
    bohr_coords = [[0, 0, -units.BOHR_ANGSTROM], [0, 0, units.BOHR_ANGSTROM]]
    np.testing.assert_allclose([atom.coords for atom in cfg.system.molecule],
                               bohr_coords)
    # Assert that the alpha and beta electrons are the same
    self.assertEqual(mol.nelec, cfg.system.electrons)
    # Assert that the basis are the same
    self.assertEqual(mol.basis, cfg.pretrain.basis)
    # Assert that atom symbols are the same
    self.assertEqual([mol.atom_symbol(i) for i in range(mol.natm)],
                     [atom.symbol for atom in cfg.system.molecule])
    # Assert that atom coordinates are the same
    pyscf_coords = [mol.atom_coords()[i] for i in range(mol.natm)]
    internal_coords = [np.array(atom.coords) for atom in cfg.system.molecule]
    np.testing.assert_allclose(pyscf_coords, internal_coords)


if __name__ == '__main__':
  absltest.main()
