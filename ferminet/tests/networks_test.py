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
"""Tests for ferminet.networks."""

from absl.testing import parameterized
from ferminet import networks
from ferminet import scf
from ferminet.utils import system
import numpy as np
import pyscf
import tensorflow.compat.v1 as tf


class NetworksTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(NetworksTest, self).setUp()
    # disable use of temp directory in pyscf.
    # Test calculations are small enough to fit in RAM and we don't need
    # checkpoint files.
    pyscf.lib.param.TMPDIR = None

  @parameterized.parameters(
      {
          'hidden_units': [8, 8],
          'after_det': [8, 8, 2],
      },
      {
          'hidden_units': [[8, 8], [8, 8]],
          'after_det': [8, 8, 2],
      },
  )
  def test_ferminet(self, hidden_units, after_det):
    """Check that FermiNet is actually antisymmetric."""
    atoms = [
        system.Atom(symbol='H', coords=(0, 0, -1.0)),
        system.Atom(symbol='H', coords=(0, 0, 1.0)),
    ]
    ne = (3, 2)
    x1 = tf.random_normal([3, 3 * sum(ne)])
    xs = tf.split(x1, sum(ne), axis=1)

    # swap indices to test antisymmetry
    x2 = tf.concat([xs[1], xs[0]] + xs[2:], axis=1)
    x3 = tf.concat(
        xs[:ne[0]] + [xs[ne[0] + 1], xs[ne[0]]] + xs[ne[0] + 2:], axis=1)

    ferminet = networks.FermiNet(
        atoms=atoms,
        nelectrons=ne,
        slater_dets=4,
        hidden_units=hidden_units,
        after_det=after_det)
    y1 = ferminet(x1)
    y2 = ferminet(x2)
    y3 = ferminet(x3)
    with tf.train.MonitoredSession() as session:
      out1, out2, out3 = session.run([y1, y2, y3])
      np.testing.assert_allclose(out1, -out2, rtol=4.e-5, atol=1.e-6)
      np.testing.assert_allclose(out1, -out3, rtol=4.e-5, atol=1.e-6)

  def test_ferminet_mask(self):
    """Check that FermiNet with a decaying mask on the output works."""
    atoms = [
        system.Atom(symbol='H', coords=(0, 0, -1.0)),
        system.Atom(symbol='H', coords=(0, 0, 1.0)),
    ]
    ne = (3, 2)
    hidden_units = [[8, 8], [8, 8]]
    after_det = [8, 8, 2]
    x = tf.random_normal([3, 3 * sum(ne)])

    ferminet = networks.FermiNet(
        atoms=atoms,
        nelectrons=ne,
        slater_dets=4,
        hidden_units=hidden_units,
        after_det=after_det,
        envelope=True)
    y = ferminet(x)
    with tf.train.MonitoredSession() as session:
      session.run(y)

  @parameterized.parameters(1, 3)
  def test_ferminet_size(self, size):
    atoms = [
        system.Atom(symbol='H', coords=(0, 0, -1.0)),
        system.Atom(symbol='H', coords=(0, 0, 1.0)),
    ]
    ne = (size, size)
    nout = 1
    batch_size = 3
    ndet = 4
    hidden_units = [[8, 8], [8, 8]]
    after_det = [8, 8, nout]
    x = tf.random_normal([batch_size, 3 * sum(ne)])

    ferminet = networks.FermiNet(
        atoms=atoms,
        nelectrons=ne,
        slater_dets=ndet,
        hidden_units=hidden_units,
        after_det=after_det)
    y = ferminet(x)
    for i in range(len(ne)):
      self.assertEqual(ferminet._dets[i].shape.as_list(), [batch_size, ndet])
      self.assertEqual(ferminet._orbitals[i].shape.as_list(),
                       [batch_size, ndet, size, size])
    self.assertEqual(y.shape.as_list(), [batch_size, nout])

  @parameterized.parameters(
      {
          'hidden_units': [8, 8],
          'after_det': [8, 8, 2]
      },
      {
          'hidden_units': [[8, 8], [8, 8]],
          'after_det': [8, 8, 2]
      },
  )
  def test_ferminet_pretrain(self, hidden_units, after_det):
    """Check that FermiNet pretraining runs."""
    atoms = [
        system.Atom(symbol='Li', coords=(0, 0, -1.0)),
        system.Atom(symbol='Li', coords=(0, 0, 1.0)),
    ]
    ne = (4, 2)
    x = tf.random_normal([1, 10, 3 * sum(ne)])

    strategy = tf.distribute.get_strategy()

    with strategy.scope():
      ferminet = networks.FermiNet(
          atoms=atoms,
          nelectrons=ne,
          slater_dets=4,
          hidden_units=hidden_units,
          after_det=after_det,
          pretrain_iterations=10)

    # Test Hartree fock pretraining - no change of position.
    hf_approx = scf.Scf(atoms, nelectrons=ne)
    hf_approx.run()
    pretrain_op_hf = networks.pretrain_hartree_fock(ferminet, x, strategy,
                                                    hf_approx)
    self.assertEqual(ferminet.pretrain_iterations, 10)
    with tf.train.MonitoredSession() as session:
      for _ in range(ferminet.pretrain_iterations):
        session.run(pretrain_op_hf)


class LogDetMatmulTest(tf.test.TestCase, parameterized.TestCase):

  def _make_mats(self, n, m, l1, l2, k, singular=False):
    x1 = np.random.randn(n, m, l1, l1).astype(np.float32)
    if singular:
      # Make one matrix singular
      x1[0, 0, 0] = 2.0 * x1[0, 0, 1]
    x2 = np.random.randn(n, m, l2, l2).astype(np.float32)
    w = np.random.randn(m, k).astype(np.float32)

    x1_tf = tf.constant(x1)
    x2_tf = tf.constant(x2)
    w_tf = tf.constant(w)

    output = networks.logdet_matmul(x1_tf, x2_tf, w_tf)
    return x1, x2, w, x1_tf, x2_tf, w_tf, output

  def _cofactor(self, x):
    u, s, v = np.linalg.svd(x)
    ss = np.tile(s[..., None], [1, s.shape[0]])
    np.fill_diagonal(ss, 1.0)
    z = np.prod(ss, axis=0)
    return np.dot(u, np.dot(np.diag(z), v)).transpose()

  @parameterized.parameters(True, False)
  def test_logdet_matmul(self, singular):
    n = 4
    m = 5
    l1 = 3
    l2 = 4
    k = 2
    x1, x2, w, _, _, _, output = self._make_mats(
        n, m, l1, l2, k, singular=singular)

    with tf.Session() as session:
      logout, signout = session.run(output)
    tf_result = np.exp(logout) * signout
    det1 = np.zeros([n, m], dtype=np.float32)
    det2 = np.zeros([n, m], dtype=np.float32)
    for i in range(n):
      for j in range(m):
        det1[i, j] = np.linalg.det(x1[i, j])
        det2[i, j] = np.linalg.det(x2[i, j])
    np_result = np.dot(det1 * det2, w)
    np.testing.assert_allclose(np_result, tf_result, atol=1e-5, rtol=1e-5)

  @parameterized.parameters(True, False)
  def test_logdet_matmul_grad(self, singular):
    n = 4
    m = 5
    l1 = 3
    l2 = 4
    k = 2
    result = self._make_mats(n, m, l1, l2, k, singular=singular)
    x1, x2, w, x1_tf, x2_tf, w_tf, output = result

    with tf.Session():
      theor_x1, numer_x1 = tf.test.compute_gradient(
          x1_tf, [n, m, l1, l1], output[0], [n, k], x_init_value=x1)
      np.testing.assert_allclose(theor_x1, numer_x1, atol=1e-2, rtol=1e-3)

      theor_x2, numer_x2 = tf.test.compute_gradient(
          x2_tf, [n, m, l2, l2], output[0], [n, k], x_init_value=x2)
      np.testing.assert_allclose(theor_x2, numer_x2, atol=1e-2, rtol=1e-3)

      theor_w, numer_w = tf.test.compute_gradient(
          w_tf, [m, k], output[0], [n, k], x_init_value=w)
      np.testing.assert_allclose(theor_w, numer_w, atol=1e-2, rtol=1e-3)

  @parameterized.parameters(True, False)
  def test_logdet_matmul_grad_grad(self, singular):
    n = 2
    m = 3
    l1 = 2
    l2 = 3
    k = 2
    result = self._make_mats(n, m, l1, l2, k, singular=singular)
    x1, x2, w, x1_tf, x2_tf, w_tf, output = result
    glog = np.random.randn(*(output[0].shape.as_list())).astype(np.float32)
    glog_tf = tf.constant(glog)
    grad_op = tf.gradients(output[0], [x1_tf, x2_tf, w_tf], grad_ys=glog_tf)
    inp_op = [x1_tf, x2_tf, w_tf, glog_tf]
    inp_np = [x1, x2, w, glog]

    with tf.Session():
      for i in range(len(inp_op)):
        for j in range(len(grad_op)):
          theor, numer = tf.test.compute_gradient(
              inp_op[i],
              inp_op[i].shape,
              grad_op[j],
              grad_op[j].shape,
              x_init_value=inp_np[i])
          np.testing.assert_allclose(theor, numer, atol=1e-2, rtol=1.3e-3)


class AdjugateGradTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (2, False),
      (2, True),
      (8, False),
      (8, True),
  )
  def test_grad_adj(self, dim, singular):

    def cofactor_tf(xs):
      s, u, v = tf.linalg.svd(xs)
      return networks.cofactor(u, s, v)

    @tf.custom_gradient
    def cofactor_tf_custom(xs):
      s, u, v = tf.linalg.svd(xs)

      def grad(dy):
        r = networks.rho(s, 0.0)
        return networks.grad_cofactor(u, v, r, dy)

      return networks.cofactor(u, s, v), grad

    n, m = 6, 3
    x = np.random.randn(n, m, dim, dim).astype(np.float32)
    if singular:
      # Make one matrix singular
      x[0, 0, 0] = 2.0 * x[0, 0, 1]
    x_tf = tf.constant(x)
    adj1 = cofactor_tf(x_tf)
    grad_adj_backprop = tf.gradients(adj1, x_tf)
    adj2 = cofactor_tf_custom(x_tf)
    grad_adj_closed = tf.gradients(adj2, x_tf)

    with tf.train.MonitoredSession() as session:
      backprop, closed = session.run([grad_adj_backprop, grad_adj_closed])

    np.testing.assert_allclose(backprop, closed, atol=1.e-5, rtol=1.e-3)


class GammaTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ((5,), 0),
      ((5,), 2),
      ((
          12,
          6,
          5,
      ), 0),
      ((
          12,
          1,
          8,
      ), 2),
      ((
          12,
          8,
          8,
      ), 2),
  )
  def test_gamma(self, shape, nsingular):

    s = tf.Variable(tf.random_uniform(shape))
    if nsingular > 0:
      zeros = list(shape)
      zeros[-1] = nsingular
      s = s[..., shape[-1] - nsingular:].assign(tf.zeros(zeros))

    g = networks.gamma(s)

    with tf.train.MonitoredSession() as session:
      s_, g_ = session.run([s, g])

    s_flat = np.reshape(s_, (-1, shape[-1]))
    gamma_exact = np.ones_like(s_flat)
    for i, row in enumerate(s_flat):
      for j in range(shape[-1]):
        row[j], r_j = 1.0, row[j]
        gamma_exact[i, j] = np.prod(row)
        row[j] = r_j
    gamma_exact = np.reshape(gamma_exact, g_.shape)
    np.testing.assert_allclose(g_, gamma_exact, atol=1.e-7)


class RhoTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (None, 2, 0),
      (None, 5, 0),
      (None, 5, 2),
      ((8, 4), 5, 2),
  )
  def test_rho(self, tile, dim, nsingular):

    assert dim > 0
    assert dim > nsingular
    s = tf.Variable(tf.random_uniform((dim,)))
    s = s[dim - nsingular:].assign(tf.zeros(nsingular))

    s_batch = s
    if tile:
      for _ in range(len(tile)):
        s_batch = tf.expand_dims(s_batch, 0)
      s_batch = tf.tile(s_batch, list(tile) + [1])

    r = networks.rho(s)
    r_batch = networks.rho(s_batch)

    with tf.train.MonitoredSession() as session:
      s_, r_, r_batch_ = session.run([s, r, r_batch])

    rho_exact = np.zeros((dim, dim), dtype=np.float32)
    for i in range(dim - 1):
      s_[i], s_i = 1.0, s_[i]
      for j in range(i + 1, dim):
        s_[j], s_j = 1.0, s_[j]
        rho_exact[i, j] = np.prod(s_)
        s_[j] = s_j
      s_[i] = s_i
    rho_exact = rho_exact + np.transpose(rho_exact)

    atol = 1e-5
    rtol = 1e-5
    np.testing.assert_allclose(r_, rho_exact, atol=atol, rtol=rtol)

    if tile:
      r_batch_ = np.reshape(r_batch_, (-1, dim, dim))
      for i in range(r_batch_[0].shape[0]):
        np.testing.assert_allclose(r_batch_[i], rho_exact, atol=atol, rtol=rtol)


if __name__ == '__main__':
  tf.test.main()
