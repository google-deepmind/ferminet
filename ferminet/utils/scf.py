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

"""Interaction with Hartree-Fock solver in pyscf."""

# Abbreviations used:
# SCF: self-consistent field (method). Another name for Hartree-Fock
# HF: Hartree-Fock method.
# RHF: restricted Hartre-Fock. Require molecular orbital for the i-th alpha-spin
#   and i-th beta-spin electrons to have the same spatial component.
# ROHF: restricted open-shell Hartree-Fock. Same as RHF except allows the number
#   of alpha and beta electrons to differ.
# UHF: unrestricted Hartre-Fock. Permits breaking of spin symmetry and hence
#   alpha and beta electrons to have different spatial components.
# AO: Atomic orbital. Underlying basis set (typically Gaussian-type orbitals and
#   built into pyscf).
# MO: molecular orbitals/Hartree-Fock orbitals. Single-particle orbitals which
#   are solutions to the Hartree-Fock equations.


from typing import Sequence, Tuple, Optional

from absl import logging
from ferminet.utils import system
import numpy as np
import pyscf


class Scf:
  """Helper class for running Hartree-Fock (self-consistent field) with pyscf.

  Attributes:
    molecule: list of system.Atom objects giving the atoms in the
      molecule and their positions.
    nelectrons: Tuple with number of alpha electrons and beta
      electrons.
    basis: Basis set to use, best specified with the relevant string
      for a built-in basis set in pyscf. A user-defined basis set can be used
      (advanced). See https://sunqm.github.io/pyscf/gto.html#input-basis for
        more details.
    pyscf_mol: the PySCF 'Molecule'. If this is passed to the init,
      the molecule, nelectrons, and basis will not be used, and the
      calculations will be performed on the existing pyscf_mol
    restricted: If true, use the restriced Hartree-Fock method, otherwise use
      the unrestricted Hartree-Fock method.
  """

  def __init__(self,
               molecule: Optional[Sequence[system.Atom]] = None,
               nelectrons: Optional[Tuple[int, int]] = None,
               basis: Optional[str] = 'cc-pVTZ',
               pyscf_mol: Optional[pyscf.gto.Mole] = None,
               restricted: bool = True):
    if pyscf_mol:
      self._mol = pyscf_mol
    else:
      self.molecule = molecule
      self.nelectrons = nelectrons
      self.basis = basis
      self._spin = nelectrons[0] - nelectrons[1]
      self._mol = None

    self.restricted = restricted
    self._mean_field = None

    pyscf.lib.param.TMPDIR = None

  def run(self):
    """Runs the Hartree-Fock calculation.

    Returns:
      A pyscf scf object (i.e. pyscf.scf.rhf.RHF, pyscf.scf.uhf.UHF or
      pyscf.scf.rohf.ROHF depending on the spin and restricted settings).

    Raises:
      RuntimeError: If the number of electrons in the PySCF molecule is not
      consistent with self.nelectrons.
    """
    # If not passed a pyscf molecule, create one
    if not self._mol:
      if any(atom.atomic_number - atom.charge > 1.e-8
             for atom in self.molecule):
        logging.info(
            'Fractional nuclear charge detected. '
            'Running SCF on atoms with integer charge.'
        )

      nuclear_charge = sum(atom.atomic_number for atom in self.molecule)
      charge = nuclear_charge - sum(self.nelectrons)
      self._mol = pyscf.gto.Mole(
          atom=[[atom.symbol, atom.coords] for atom in self.molecule],
          unit='bohr')
      self._mol.basis = self.basis
      self._mol.spin = self._spin
      self._mol.charge = charge
      self._mol.build()
      if self._mol.nelectron != sum(self.nelectrons):
        raise RuntimeError('PySCF molecule not consistent with QMC molecule.')
    if self.restricted:
      self._mean_field = pyscf.scf.RHF(self._mol)
    else:
      self._mean_field = pyscf.scf.UHF(self._mol)
    self._mean_field.init_guess = 'atom'
    self._mean_field.kernel()
    return self._mean_field

  def eval_mos(self, positions: np.ndarray,
               deriv: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluates the Hartree-Fock single-particle orbitals at a set of points.

    Args:
      positions: numpy array of shape (N, 3) of the positions in space at which
        to evaluate the Hartree-Fock orbitals.
      deriv: If True, also calculate the first derivatives of the
        single-particle orbitals.

    Returns:
      Pair of numpy float64 arrays of shape (N, M) (deriv=False) or (4, N, M)
      (deriv=True), where 2M is the number of Hartree-Fock orbitals. The (i-th,
      j-th) element in the first (second) array gives the value of the j-th
      alpha (beta) Hartree-Fock orbital at the i-th electron position in
      positions. For restricted (RHF, ROHF) calculations, the two arrays will be
      identical.
      If deriv=True, the first index contains [value, x derivative, y
      derivative, z derivative].

    Raises:
      RuntimeError: If Hartree-Fock calculation has not been performed using
        `run`.
      NotImplementedError: If Hartree-Fock calculation used Cartesian
        Gaussian-type orbitals as the underlying basis set.
    """
    if self._mean_field is None:
      raise RuntimeError('Mean-field calculation has not been run.')
    if self.restricted:
      coeffs = (self._mean_field.mo_coeff,)
    else:
      coeffs = self._mean_field.mo_coeff
    # Assumes self._mol.cart (use of Cartesian Gaussian-type orbitals and
    # integrals) is False (default behaviour of pyscf).
    if self._mol.cart:
      raise NotImplementedError(
          'Evaluation of molecular orbitals using cartesian GTOs.')
    # Note sph refers to the use of spherical GTO basis sets rather than
    # Cartesian GO basis sets. The coordinate system used for the electron
    # positions is Cartesian in both cases.
    gto_op = 'GTOval_sph_deriv1' if deriv else 'GTOval_sph'
    ao_values = self._mol.eval_gto(gto_op, positions)
    mo_values = tuple(np.matmul(ao_values, coeff) for coeff in coeffs)
    if self.restricted:
      # duplicate for beta electrons.
      mo_values *= 2
    return mo_values
