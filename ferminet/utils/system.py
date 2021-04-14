# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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

"""Functions to create different kinds of systems."""

from typing import Sequence
import attr
from ferminet.utils import elements
from ferminet.utils import units as unit_conversion
import numpy as np


# Default bond lengths in angstrom for some diatomics.
# Bond lengths from either the G3 dataset:
# 1. http://www.cse.anl.gov/OldCHMwebsiteContent/compmat/comptherm.htm
# 2. L. A. Curtiss, P. C. Redfern, K. Raghavachari, and J. A. Pople,
#    J. Chem. Phys, 109, 42 (1998).
# or from NIST (https://cccbdb.nist.gov/diatomicexpbondx.asp).
diatomic_bond_lengths = {
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
    'BH': 1.2324,
    'PN': 1.491,
    'AlH': 1.648,
    'AlN': 1.786,
}


# Default spin polarisation for a few diatomics of interest.
# Otherwise default to either singlet (doublet) for even (odd) numbers of
# electrons. Units: number of unpaired electrons.
diatomic_spin_polarisation = {
    'B2': 2,
    'O2': 2,
    'NH': 2,
    'AlN': 2,
}


@attr.s
class Atom:  # pytype: disable=invalid-function-definition
  """Atom information for Hamiltonians.

  The nuclear charge is inferred from the symbol if not given, in which case the
  symbol must be the IUPAC symbol of the desired element.

  Attributes:
    symbol: Element symbol.
    coords: An iterable of atomic coordinates. Always a list of floats and in
      bohr after initialisation. Default: place atom at origin.
    charge: Nuclear charge. Default: nuclear charge (atomic number) of atom of
      the given name.
    atomic_number: Atomic number associated with element. Default: atomic number
      of element of the given symbol. Should match charge unless fractional
      nuclear charges are being used.
    units: String giving units of coords. Either bohr or angstrom. Default:
      bohr. If angstrom, coords are converted to be in bohr and units to the
      string 'bohr'.
    coords_angstrom: list of atomic coordinates in angstrom.
    coords_array: Numpy array of atomic coordinates in bohr.
    element: elements.Element corresponding to the symbol.
  """
  symbol = attr.ib()
  coords = attr.ib(
      converter=lambda xs: tuple(float(x) for x in xs),
      default=(0.0, 0.0, 0.0))  # type: Sequence[float]
  charge = attr.ib(converter=float)
  atomic_number = attr.ib(converter=int)
  units = attr.ib(
      default='bohr', validator=attr.validators.in_(['bohr', 'angstrom']))

  @charge.default
  def _set_default_charge(self):
    return self.element.atomic_number

  @atomic_number.default
  def _set_default_atomic_number(self):
    return self.element.atomic_number

  def __attrs_post_init__(self):
    if self.units == 'angstrom':
      self.coords = [unit_conversion.angstrom2bohr(x) for x in self.coords]
      self.units = 'bohr'

  @property
  def coords_angstrom(self):
    return [unit_conversion.bohr2angstrom(x) for x in self.coords]

  @property
  def coords_array(self):
    if not hasattr(self, '_coords_arr'):
      self._coords_arr = np.array(self.coords)
    return self._coords_arr

  @property
  def element(self):
    return elements.SYMBOLS[self.symbol]


def atom(symbol, spins=None, charge=0):
  """Return configuration for a single atom.

  Args:
    symbol: The atomic symbol from the periodic table
    spins (optional): A tuple with the number of spin-up and spin-down electrons
    charge (optional): If zero (default), create a neutral atom, otherwise
      create an anion if charge is negative or cation if charge is positive.
  Returns:
    A list with a single Atom object located at zero, and a tuple with the spin
    configuration of the electrons.
  """
  atomic_number = elements.SYMBOLS[symbol].atomic_number
  if charge > atomic_number:
    raise ValueError('Cannot have a cation with charge larger than the '
                     'atomic number. Charge: {}, Atomic Number{}'.format(
                         charge, atomic_number))
  if spins is None:
    spin_polarisation = elements.ATOMIC_NUMS[atomic_number-charge].spin_config
    nalpha = (atomic_number + spin_polarisation) // 2
    spins = (nalpha, atomic_number - charge - nalpha)
  return [Atom(symbol=symbol, coords=(0.0, 0.0, 0.0))], spins


def diatomic(symbol1, symbol2, bond_length, spins=None, charge=0, units='bohr'):
  """Return configuration for a diatomic molecule."""
  if spins is None:
    atomic_number_1 = elements.SYMBOLS[symbol1].atomic_number
    atomic_number_2 = elements.SYMBOLS[symbol2].atomic_number
    total_charge = atomic_number_1 + atomic_number_2 - charge
    if total_charge % 2 == 0:
      spins = (total_charge // 2, total_charge // 2)
    else:
      spins = ((total_charge + 1)// 2, (total_charge - 1) // 2)

  return [
      Atom(symbol=symbol1, coords=(0.0, 0.0, bond_length/2.0), units=units),
      Atom(symbol=symbol2, coords=(0.0, 0.0, -bond_length/2.0), units=units)
  ], spins


def molecule(symbol, bond_length=0.0, units='bohr'):
  """Hardcoded molecular geometries from the original Fermi Net paper."""

  if symbol in diatomic_bond_lengths:
    if symbol[-1] == '2':
      symbs = [symbol[:-1], symbol[:-1]]
    else:  # Split a camel-case string on the second capital letter
      split_idx = None
      for i in range(1, len(symbol)):
        if split_idx is None and symbol[i].isupper():
          split_idx = i
      if split_idx is None:
        raise ValueError('Cannot find second atomic symbol: {}'.format(symbol))
      symbs = [symbol[:split_idx], symbol[split_idx:]]

    atomic_number_1 = elements.SYMBOLS[symbs[0]].atomic_number
    atomic_number_2 = elements.SYMBOLS[symbs[1]].atomic_number
    total_charge = atomic_number_1 + atomic_number_2
    if symbol in diatomic_spin_polarisation:
      spin_pol = diatomic_spin_polarisation[symbol]
      spins = ((total_charge + spin_pol) // 2, (total_charge + spin_pol) // 2)
    elif total_charge % 2 == 0:
      spins = (total_charge // 2, total_charge // 2)
    else:
      spins = ((total_charge + 1)// 2, (total_charge - 1) // 2)

    if bond_length == 0.0:
      bond_length = diatomic_bond_lengths[symbol]
      units = 'angstrom'
    return diatomic(symbs[0], symbs[1],
                    bond_length,
                    units=units,
                    spins=spins)

  if bond_length != 0.0:
    raise ValueError('Bond length argument only appropriate for diatomics.')

  if symbol == 'CH4':
    return [
        Atom(symbol='C', coords=(0.0, 0.0, 0.0), units='bohr'),
        Atom(symbol='H', coords=(1.18886, 1.18886, 1.18886), units='bohr'),
        Atom(symbol='H', coords=(-1.18886, -1.18886, 1.18886), units='bohr'),
        Atom(symbol='H', coords=(1.18886, -1.18886, -1.18886), units='bohr'),
        Atom(symbol='H', coords=(-1.18886, 1.18886, -1.18886), units='bohr'),
    ], (5, 5)

  if symbol == 'NH3':
    return [
        Atom(symbol='N', coords=(0.0, 0.0, 0.22013), units='bohr'),
        Atom(symbol='H', coords=(0.0, 1.77583, -0.51364), units='bohr'),
        Atom(symbol='H', coords=(1.53791, -0.88791, -0.51364), units='bohr'),
        Atom(symbol='H', coords=(-1.53791, -0.88791, -0.51364), units='bohr'),
    ], (5, 5)

  if symbol in ('C2H4', 'ethene', 'ethylene'):
    return [
        Atom(symbol='C', coords=(0.0, 0.0, 1.26135), units='bohr'),
        Atom(symbol='C', coords=(0.0, 0.0, -1.26135), units='bohr'),
        Atom(symbol='H', coords=(0.0, 1.74390, 2.33889), units='bohr'),
        Atom(symbol='H', coords=(0.0, -1.74390, 2.33889), units='bohr'),
        Atom(symbol='H', coords=(0.0, 1.74390, -2.33889), units='bohr'),
        Atom(symbol='H', coords=(0.0, -1.74390, -2.33889), units='bohr'),
    ], (8, 8)

  if symbol in ('C4H6', 'bicyclobutane'):
    return [
        Atom(symbol='C', coords=(0.0, 2.13792, 0.58661), units='bohr'),
        Atom(symbol='C', coords=(0.0, -2.13792, 0.58661), units='bohr'),
        Atom(symbol='C', coords=(1.41342, 0.0, -0.58924), units='bohr'),
        Atom(symbol='C', coords=(-1.41342, 0.0, -0.58924), units='bohr'),
        Atom(symbol='H', coords=(0.0, 2.33765, 2.64110), units='bohr'),
        Atom(symbol='H', coords=(0.0, 3.92566, -0.43023), units='bohr'),
        Atom(symbol='H', coords=(0.0, -2.33765, 2.64110), units='bohr'),
        Atom(symbol='H', coords=(0.0, -3.92566, -0.43023), units='bohr'),
        Atom(symbol='H', coords=(2.67285, 0.0, -2.19514), units='bohr'),
        Atom(symbol='H', coords=(-2.67285, 0.0, -2.19514), units='bohr'),
    ], (15, 15)

  raise ValueError('Not a recognized molecule: {}'.format(symbol))


def hn(n, r, charge=0, units='bohr'):
  """Return a hydrogen chain with n atoms and separation r."""
  m = n - charge  # number of electrons
  if m % 2 == 0:
    spins = (m//2, m//2)
  else:
    spins = ((m+1)//2, (m-1)//2)
  lim = r * (n-1) / 2.0
  return [Atom(symbol='H', coords=(0.0, 0.0, z), units=units)
          for z in np.linspace(-lim, lim, n)], spins


def h4_circle(r, theta, units='bohr'):
  """Return 4 hydrogen atoms arranged in a circle, a failure case of CCSD(T)."""
  return [
      Atom(symbol='H',
           coords=(r*np.cos(theta), r*np.sin(theta), 0.0),
           units=units),
      Atom(symbol='H',
           coords=(-r*np.cos(theta), r*np.sin(theta), 0.0),
           units=units),
      Atom(symbol='H',
           coords=(r*np.cos(theta), -r*np.sin(theta), 0.0),
           units=units),
      Atom(symbol='H',
           coords=(-r*np.cos(theta), -r*np.sin(theta), 0.0),
           units=units)
  ], (2, 2)
