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

"""Tests for gto.py."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet.utils import gto
import jax
import jax.numpy as jnp
import numpy as np
import pyscf.dft
import pyscf.gto
import pyscf.lib


class GtoTest(parameterized.TestCase):

  def setUp(self):
    super(GtoTest, self).setUp()
    pyscf.lib.param.TMPDIR = None
    pyscf.lib.num_threads(1)

  @parameterized.parameters([['a', True], ['a', False], ['b', False]])
  def test_eval_gto(self, unit, jit):
    mol = pyscf.gto.M(atom='Na 0 0 -1; F 0 0 1', basis='def2-qzvp', unit=unit)
    solver = pyscf.dft.RKS(mol)
    solver.grids.build()
    coords = solver.grids.coords

    aos_pyscf = mol.eval_gto('GTOval_sph', coords)

    mol_jax = gto.Mol.from_pyscf_mol(mol)
    jax_eval_gto = jax.jit(mol_jax.eval_gto) if jit else mol_jax.eval_gto
    aos_jax = jax_eval_gto(coords)

    # Loose tolerances due to float32. With float64, these agree to better than
    # 1e-10.
    np.testing.assert_allclose(aos_pyscf, aos_jax, atol=4.e-4, rtol=2.e-4)

  def test_grad_solid_harmonic(self):
    np.random.seed(0)
    r = np.random.randn(100, 3)
    l_max = 5

    jax_grad = jax.jacfwd(lambda x: gto.solid_harmonic_from_cart(x, l_max))
    expected = jnp.transpose(jnp.squeeze(jax.vmap(jax_grad)(r[:, None, :])),
                             [1, 2, 0, 3])

    with self.subTest('by hand'):
      observed = gto.grad_solid_harmonic(r, l_max)
      np.testing.assert_allclose(observed, expected, atol=1.e-4)
    with self.subTest('by jax'):
      observed_jacfwd = gto.grad_solid_harmonic_by_jacfwd(r, l_max)
      np.testing.assert_allclose(observed_jacfwd, expected, atol=1.e-4)

if __name__ == '__main__':
  absltest.main()
