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

"""Tests for ferminet.utils.scf."""

from typing import List, Tuple
from absl.testing import absltest
from absl.testing import parameterized
from ferminet.utils import scf
from ferminet.utils import system
import numpy as np
import pyscf


class ScfTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # disable use of temp directory in pyscf.
    # Test calculations are small enough to fit in RAM and we don't need
    # checkpoint files.
    pyscf.lib.param.TMPDIR = None

  @parameterized.parameters(
      {
          'molecule': [system.Atom('He', (0, 0, 0))],
          'nelectrons': (1, 1)
      },
      {
          'molecule': [system.Atom('N', (0, 0, 0))],
          'nelectrons': (5, 2)
      },
      {
          'molecule': [system.Atom('N', (0, 0, 0))],
          'nelectrons': (5, 3)
      },
      {
          'molecule': [system.Atom('N', (0, 0, 0))],
          'nelectrons': (4, 2)
      },
      {
          'molecule': [system.Atom('O', (0, 0, 0))],
          'nelectrons': (5, 3),
          'restricted': False,
      },
      {
          'molecule': [
              system.Atom('N', (0, 0, 0)),
              system.Atom('N', (0, 0, 1.4))
          ],
          'nelectrons': (7, 7)
      },
      {
          'molecule': [
              system.Atom('O', (0, 0, 0)),
              system.Atom('O', (0, 0, 1.4))
          ],
          'nelectrons': (9, 7),
          'restricted': False,
      },
  )
  def test_scf_interface(self,
                         molecule: List[system.Atom],
                         nelectrons: Tuple[int, int],
                         restricted: bool = True):
    """Tests SCF interface to a pyscf calculation.

    pyscf has its own tests so only check that we can run calculations over
    atoms and simple diatomics using the interface in ferminet.scf.

    Args:
      molecule: List of system.Atom objects giving atoms in the molecule.
      nelectrons: Tuple containing number of alpha and beta electrons.
      restricted: If true, run a restricted Hartree-Fock calculation, otherwise
        run an unrestricted Hartree-Fock calculation.
    """
    npts = 100
    xs = np.random.randn(npts, 3)
    hf = scf.Scf(molecule=molecule,
                 nelectrons=nelectrons,
                 restricted=restricted)
    hf.run()
    mo_vals = hf.eval_mos(xs)
    self.assertLen(mo_vals, 2)  # alpha-spin orbitals and beta-spin orbitals.
    for spin_mo_vals in mo_vals:
      # Evaluate npts points on M orbitals/functions - (npts, M) array.
      self.assertEqual(spin_mo_vals.shape, (npts, hf._mol.nao_nr()))


if __name__ == '__main__':
  absltest.main()
