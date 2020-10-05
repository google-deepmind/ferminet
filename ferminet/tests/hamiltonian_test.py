# Lint as: python3
# Copyright 2018 DeepMind Technologies Limited. All Rights Reserved.
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

"""Tests for ferminet.hamiltonian."""

from ferminet import hamiltonian
from ferminet.utils import system
import numpy as np
import tensorflow.compat.v1 as tf


def _hydrogen(xs):
  """Hydrogen (3D) atom ground state (1s) wavefunction.

  Energy: -1/2 hartrees.

  Args:
    xs: tensor (batch_size, 3) of electron positions.

  Returns:
    tensor (batch_size, 1) of the (unnormalised) wavefunction at each position.
  """
  with tf.name_scope('Psi_H'):
    return tf.exp(-tf.norm(xs, axis=1, keepdims=True))


def _helium(xs):
  """Compact parametrized Helium wavefunction.

  See https://opencommons.uconn.edu/chem_educ/30/ and Phys Rev A 74, 014501
  (2006).

  Energy: -2.901188 hartrees (compared to -2.9037243770 hartrees for the exact
  ground state).

  Args:
    xs: tensor (batch_size, 6) of electron positions.

  Returns:
    tensor (batch_size, 1) of the (unnormalised) wavefunction at each pair of
    positions.
  """
  with tf.name_scope('Psi_He'):
    x1, x2 = tf.split(xs, 2, axis=1)
    x1n = tf.norm(x1, axis=1, keepdims=True)
    x2n = tf.norm(x2, axis=1, keepdims=True)
    s = x1n + x2n
    t = x1n - x2n
    u = tf.norm(x1 - x2, axis=1, keepdims=True)
    return (tf.exp(-2*s)
            * (1 + 0.5*u*tf.exp(-1.013*u))
            * (1 + 0.2119*s*u + 0.1406*t*t - 0.003*u*u))


def _grid_points(start, stop, npoints, ndim):
  x1 = np.linspace(start, stop, npoints)
  x2 = np.zeros(npoints)
  g1 = np.array([x1] + [x2 for _ in range(ndim-1)]).T
  g2 = np.array([x1, x1] + [x2 for _ in range(ndim-2)]).T
  return np.ascontiguousarray(g1), np.ascontiguousarray(g2)


def _wfn_to_log(f):
  def _logf(x):
    psi = f(x)
    return tf.log(tf.abs(psi)), tf.sign(psi)
  return _logf


class WavefunctionTest(tf.test.TestCase):
  """Basic tests for the toy wavefunctions defined above."""

  def test_hydrogen(self):
    grid1, grid2 = _grid_points(-1.1, 1.1, 21, 3)
    psi_np = np.exp(-np.abs(grid1[:, 0]))

    psi = _hydrogen(tf.constant(grid1))
    with tf.train.MonitoredSession() as session:
      psi_ = np.squeeze(session.run(psi))
    np.testing.assert_allclose(psi_, psi_np)
    np.testing.assert_allclose(psi_, psi_[::-1])

    psi = _hydrogen(tf.constant(grid2))
    with tf.train.MonitoredSession() as session:
      psi_ = np.squeeze(session.run(psi))
    np.testing.assert_allclose(psi_, psi_np**np.sqrt(2))
    np.testing.assert_allclose(psi_, psi_[::-1])

  def test_helium(self):
    grid1, grid2 = _grid_points(-1.1, 1.0, 5, 6)
    grid1[:, 3] += 0.5  # Avoid nuclear singularity for second electron
    grid2[:, 3] += 0.5  # Avoid nuclear singularity for second electron

    # Exact results calculated by hand...
    psi1 = np.array([0.07484748391,
                     0.17087261364,
                     0.42062736263,
                     0.14476421949,
                     0.06836260905])
    psi2 = np.array([0.037059842334,
                     0.114854125198,
                     0.403704643707,
                     0.123475237250,
                     0.040216273342])

    psi = _helium(tf.constant(grid1))
    with tf.train.MonitoredSession() as session:
      psi_ = np.squeeze(session.run(psi))
    np.testing.assert_allclose(psi_, psi1)

    # Check for symmetry: swap x and z and electron 1 and electron 2.
    grid1 = np.flip(grid1, axis=1)
    psi = _helium(tf.constant(grid1))
    with tf.train.MonitoredSession() as session:
      psiz_ = np.squeeze(session.run(psi))
    np.testing.assert_allclose(psi_, psiz_)

    psi = _helium(tf.constant(grid2))
    with tf.train.MonitoredSession() as session:
      psi_ = np.squeeze(session.run(psi))
    np.testing.assert_allclose(psi_, psi2)


class HamiltonianAtomTest(tf.test.TestCase):

  def test_hydrogen_atom(self):
    xs = tf.random_uniform((6, 3), minval=-1.0, maxval=1.0)
    atoms = [system.Atom(symbol='H', coords=(0, 0, 0))]
    kin, pot = hamiltonian.operators(atoms, 1, 0.0)
    logpsi = tf.log(tf.abs(_hydrogen(xs)))
    e_loc = kin(logpsi, xs) + pot(xs)
    with tf.train.MonitoredSession() as session:
      e_loc_ = session.run(e_loc)
    # Wavefunction is the ground state eigenfunction. Hence H\Psi = -1/2 \Psi.
    np.testing.assert_allclose(e_loc_, -0.5, rtol=1.e-6, atol=1.e-7)

  def test_helium_atom(self):
    x1 = np.linspace(-1, 1, 6, dtype=np.float32)
    x2 = np.zeros(6, dtype=np.float32) + 0.25
    x12 = np.array([x1] + [x2 for _ in range(5)]).T
    atoms = [system.Atom(symbol='He', coords=(0, 0, 0))]
    hpsi_test = np.array([
        -0.25170374, -0.43359983, -0.61270618, -1.56245542, -0.39977553,
        -0.2300467
    ])

    log_helium = _wfn_to_log(_helium)
    h = hamiltonian.exact_hamiltonian(atoms, 2)
    xs = tf.constant(x12)
    _, hpsi = h(log_helium, xs)
    with tf.train.MonitoredSession() as session:
      hpsi_ = session.run(hpsi)

    xs = tf.reverse(xs, axis=[1])
    _, hpsi2 = h(log_helium, xs)
    with tf.train.MonitoredSession() as session:
      hpsi2_ = session.run(hpsi2)

    np.testing.assert_allclose(hpsi_[:, 0], hpsi_test, rtol=1.e-6)
    np.testing.assert_allclose(hpsi_, hpsi2_, rtol=1.e-6)


class HamiltonianMoleculeTest(tf.test.TestCase):

  def setUp(self):
    super(HamiltonianMoleculeTest, self).setUp()
    self.rh1 = np.array([0, 0, 0], dtype=np.float32)
    self.rh2 = np.array([0, 0, 200], dtype=np.float32)
    self.rhh = self.rh2 - self.rh1
    self.dhh = 1.0 / np.sqrt(np.dot(self.rhh, self.rhh))
    self.xs = (2 * np.random.random((6, 3)) - 1).astype(np.float32)
    self.xs = [self.xs + self.rh1, self.xs + self.rh2]
    self.atoms = [
        system.Atom(symbol='H', coords=self.rh1),
        system.Atom(symbol='H', coords=self.rh2)
    ]

  def test_hydrogen_molecular_ion(self):

    def h2_psi(xs):
      # Trial wavefunction as linear combination of H 1s wavefunctions centered
      # on each atom.
      return _hydrogen(xs - self.rh1) + _hydrogen(xs - self.rh2)

    h = hamiltonian.exact_hamiltonian(self.atoms, 1, 0.0)
    # H2+; 1 electron. Use points around both nuclei.
    xs = tf.constant(np.concatenate(self.xs, axis=0))
    psi, hpsi = h(_wfn_to_log(h2_psi), xs)
    with tf.train.MonitoredSession() as session:
      psi_, hpsi_ = session.run([psi, hpsi])
    # Leading order correction to energy is nuclear interaction with the far
    # nucleus.
    np.testing.assert_allclose(
        hpsi_, -(0.5 + self.dhh) * psi_, rtol=self.dhh, atol=self.dhh)

  def test_hydrogen_molecule(self):

    def h2_psi(xs):
      x1, x2 = tf.split(xs, 2, axis=1)
      # Essentially a non-interacting system.
      return _hydrogen(x1 - self.rh1) * _hydrogen(x2 - self.rh2)

    h = hamiltonian.exact_hamiltonian(self.atoms, 2, 0.0)
    # H2; 2 electrons. Place one electron around each nucleus.
    xs = tf.constant(np.concatenate(self.xs, axis=1))
    psi, hpsi = h(_wfn_to_log(h2_psi), xs)
    with tf.train.MonitoredSession() as session:
      psi_, hpsi_ = session.run([psi, hpsi])
    np.testing.assert_allclose(
        hpsi_, -(1 + self.dhh) * psi_, rtol=self.dhh, atol=self.dhh)


class R12FeaturesTest(tf.test.TestCase):

  def test_r12_features_atom1(self):
    atoms = [system.Atom(symbol='H', coords=(0, 0, 0))]
    one = np.ones((1, 3))
    xs = tf.constant(one, dtype=tf.float32)
    xs12 = hamiltonian.r12_features(xs, atoms, 1, flatten=True)
    with tf.train.MonitoredSession() as session:
      xs12_ = session.run(xs12)
    # Should add |x|.
    x_test = np.concatenate([[[np.linalg.norm(one[0])]], one], axis=1)
    np.testing.assert_allclose(xs12_, x_test)

  def test_r12_features_atom2(self):
    atoms = [system.Atom(symbol='He', coords=(0, 0, 0))]
    one = np.concatenate([np.ones((1, 3)), 0.5 * np.ones((1, 3))], axis=1)
    xs = tf.constant(one, dtype=tf.float32)
    xs12 = hamiltonian.r12_features(xs, atoms, 2, flatten=True)
    with tf.train.MonitoredSession() as session:
      xs12_ = session.run(xs12)
    # Should add |x_1|, |x_2|, |x1-x2|
    norm = np.linalg.norm
    x_test = np.concatenate([
        [[
            norm(one[0, :3]),
            norm(one[0, 3:]),
            norm(one[0, :3] - one[0, 3:]),
        ]],
        one
    ],
                            axis=1)
    np.testing.assert_allclose(xs12_, x_test)

  def test_r12_features_molecule(self):
    atoms = [
        system.Atom(symbol='H', coords=(0, 0, 0)),
        system.Atom(symbol='H', coords=(1, 0, 0))
    ]
    one = np.concatenate([np.ones((1, 3)), 0.5 * np.ones((1, 3))], axis=1)
    xs = tf.constant(one, dtype=tf.float32)
    xs12 = hamiltonian.r12_features(xs, atoms, 2, flatten=True)
    with tf.train.MonitoredSession() as session:
      xs12_ = session.run(xs12)
    # Should add |x_11|, |x_21|, |x_12|, |x_22|, |x1-x2|
    norm = np.linalg.norm
    x_test = np.concatenate([
        [[
            norm(one[0, :3]),
            norm(one[0, :3] - atoms[1].coords),
            norm(one[0, 3:]),
            norm(one[0, 3:] - atoms[1].coords),
            norm(one[0, :3] - one[0, 3:]),
        ]],
        one
    ],
                            axis=1)
    np.testing.assert_allclose(xs12_, x_test, rtol=1e-6)


if __name__ == '__main__':
  tf.test.main()
