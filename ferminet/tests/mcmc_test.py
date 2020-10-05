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

"""Tests for ferminet.mcmc."""

from absl.testing import parameterized

from ferminet import hamiltonian
from ferminet import mcmc
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


def _run_mcmc(atoms,
              nelectrons,
              net,
              batch_size=1024,
              steps=10,
              dtype=tf.float32):
  gen = mcmc.MCMC(
      net,
      batch_size,
      [0] * 3 * nelectrons,
      1.0,
      0.1,
      dtype=dtype)
  kin, pot = hamiltonian.operators(atoms, nelectrons, 0.0)
  walkers = tf.squeeze(gen.walkers)
  psi, _ = net(walkers)
  e_loc = tf.reduce_sum(kin(psi, walkers) + pot(walkers)) / batch_size
  e = []
  mcmc_step = gen.step()
  with tf.train.MonitoredSession() as session:
    for _ in range(steps):
      session.run(mcmc_step)
      e.append(session.run(e_loc))
  return np.array(e)


class McmcTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      {'dtype': tf.float64},
      {'dtype': tf.float64},
  )
  def test_hydrogen_atom(self, dtype):
    atoms = [system.Atom(symbol='H', coords=(0, 0, 0))]
    def net(x):
      psi = _hydrogen(x)
      return (tf.log(tf.abs(psi)), tf.sign(psi))
    e = _run_mcmc(atoms, 1, net, dtype=dtype)
    np.testing.assert_allclose(e, -np.ones_like(e) / 2)

  def test_helium_atom(self):
    atoms = [system.Atom(symbol='He', coords=(0, 0, 0))]
    def net(x):
      psi = _helium(x)
      return (tf.log(tf.abs(psi)), tf.sign(psi))
    e = _run_mcmc(atoms, 2, net, steps=500)
    np.testing.assert_allclose(e[100:].mean(), -2.901188, atol=5.e-3)

if __name__ == '__main__':
  tf.test.main()
