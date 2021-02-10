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

"""Tests for ferminet.scf."""

from absl.testing import parameterized
from ferminet import scf
from ferminet.utils import system
import numpy as np
import pyscf
import tensorflow.compat.v1 as tf


def create_o2_hf(bond_length):
  molecule = [
      system.Atom('O', (0, 0, 0)),
      system.Atom('O', (0, 0, bond_length))
  ]
  spin = 2
  oxygen_atomic_number = 8
  nelectrons = [oxygen_atomic_number + spin, oxygen_atomic_number - spin]
  hf = scf.Scf(molecule=molecule, nelectrons=nelectrons, restricted=False)
  return hf


class ScfTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(ScfTest, self).setUp()
    # disable use of temp directory in pyscf.
    # Test calculations are small enough to fit in RAM and we don't need
    # checkpoint files.
    pyscf.lib.param.TMPDIR = None

  @parameterized.parameters(np.float32, np.float64)
  def test_tf_eval_mos(self, dtype):
    """Tests tensorflow op for evaluating Hartree-Fock orbitals.

    Args:
      dtype: numpy type to use for position vectors.
    """
    xs = np.random.randn(100, 3).astype(dtype)
    hf = create_o2_hf(1.4)
    hf.run()

    # Evaluate MOs at positions xs directly in pyscf.
    # hf.eval_mos returns np.float64 arrays always. Cast to dtype for
    # like-for-like comparison.
    mo_vals = [mo_val.astype(dtype) for mo_val in hf.eval_mos(xs, deriv=True)]

    # Evaluate MOs as a tensorflow op.
    tf_xs = tf.constant(xs)
    tf_mo_vals = hf.tf_eval_mos(tf_xs, deriv=True)
    tf_dmo_vals = tf.gradients(tf.square(tf_mo_vals), tf_xs)[0]
    tf_dmo_vals_shape = tf.shape(tf_dmo_vals)
    with tf.train.MonitoredSession() as session:
      tf_mo_vals_, tf_dmo_vals_ = session.run([tf_mo_vals, tf_dmo_vals])
      tf_dmo_vals_shape_ = session.run(tf_dmo_vals_shape)
    tf_mo_vals_ = np.split(tf_mo_vals_, 2, axis=-1)

    for np_val, tf_val in zip(mo_vals, tf_mo_vals_):
      self.assertEqual(np_val.dtype, tf_val.dtype)
      np.testing.assert_allclose(np_val[0], tf_val)

    np.testing.assert_array_equal(tf_dmo_vals_shape_, xs.shape)
    np.testing.assert_array_equal(tf_dmo_vals_shape_, tf_dmo_vals_.shape)
    # Compare analytic derivative of orbital^2 with tensorflow backprop.
    # Sum over the orbital index and spin channel because of definition of
    # tensorflow gradients.
    np_deriv = sum(
        np.sum(2 * np_val[1:] * np.expand_dims(np_val[0], 0), axis=-1)
        for np_val in mo_vals)
    # Tensorflow returns (N,3) but pyscf returns (3, N, M).
    np_deriv = np.transpose(np_deriv)
    self.assertEqual(np_deriv.dtype, tf_dmo_vals_.dtype)
    if dtype == np.float32:
      atol = 1e-6
      rtol = 1e-6
    else:
      atol = 0
      rtol = 1.e-7
    np.testing.assert_allclose(np_deriv, tf_dmo_vals_, atol=atol, rtol=rtol)

  def test_tf_eval_mos_deriv(self):

    hf = create_o2_hf(1.4)
    hf.run()
    xs = tf.random_normal((100, 3))
    tf_mos_derivs = hf.tf_eval_mos(xs, deriv=True)
    tf_mos = hf.tf_eval_mos(xs, deriv=False)
    with tf.train.MonitoredSession() as session:
      tf_mos_, tf_mos_derivs_ = session.run([tf_mos, tf_mos_derivs])
    np.testing.assert_allclose(tf_mos_, tf_mos_derivs_)

  def test_tf_eval_hf(self):

    # Check we get consistent answers between multiple calls to eval_mos and
    # a single tf_eval_hf call.
    molecule = [system.Atom('O', (0, 0, 0))]
    nelectrons = (5, 3)
    hf = scf.Scf(molecule=molecule,
                 nelectrons=nelectrons,
                 restricted=False)
    hf.run()

    batch = 100
    xs = [np.random.randn(batch, 3) for _ in range(sum(nelectrons))]
    mos = []
    for i, x in enumerate(xs):
      ispin = 0 if i < nelectrons[0] else 1
      orbitals = hf.eval_mos(x)[ispin]
      # Select occupied orbitals via Aufbau.
      mos.append(orbitals[:, :nelectrons[ispin]])
    np_mos = (np.stack(mos[:nelectrons[0]], axis=1),
              np.stack(mos[nelectrons[0]:], axis=1))

    tf_xs = tf.constant(np.stack(xs, axis=1))
    tf_mos = hf.tf_eval_hf(tf_xs, deriv=True)
    with tf.train.MonitoredSession() as session:
      tf_mos_ = session.run(tf_mos)

    for i, (np_mos_mat, tf_mos_mat) in enumerate(zip(np_mos, tf_mos_)):
      self.assertEqual(np_mos_mat.shape, tf_mos_mat.shape)
      self.assertEqual(np_mos_mat.shape, (batch, nelectrons[i], nelectrons[i]))
      np.testing.assert_allclose(np_mos_mat, tf_mos_mat)

  def test_tf_eval_slog_wavefuncs(self):

    # Check TensorFlow evaluation runs and gives correct shapes.
    molecule = [system.Atom('O', (0, 0, 0))]
    nelectrons = (5, 3)
    total_electrons = sum(nelectrons)
    num_spatial_dim = 3
    hf = scf.Scf(molecule=molecule,
                 nelectrons=nelectrons,
                 restricted=False)
    hf.run()

    batch = 100
    rng = np.random.RandomState(1)
    flat_positions_np = rng.randn(batch,
                                  total_electrons * num_spatial_dim)

    flat_positions_tf = tf.constant(flat_positions_np)
    for method in [hf.tf_eval_slog_slater_determinant,
                   hf.tf_eval_slog_hartree_product]:
      slog_wavefunc, signs = method(flat_positions_tf)
      with tf.train.MonitoredSession() as session:
        slog_wavefunc_, signs_ = session.run([slog_wavefunc, signs])
      self.assertEqual(slog_wavefunc_.shape, (batch, 1))
      self.assertEqual(signs_.shape, (batch, 1))
    hartree_product = hf.tf_eval_hartree_product(flat_positions_tf)
    with tf.train.MonitoredSession() as session:
      hartree_product_ = session.run(hartree_product)
    np.testing.assert_allclose(hartree_product_,
                               np.exp(slog_wavefunc_) * signs_)


if __name__ == '__main__':
  tf.test.main()
