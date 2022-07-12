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

"""Tests for ferminet.hamiltonian."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import hamiltonian
from ferminet import networks
import jax
import jax.numpy as jnp
import numpy as np


def h_atom_log_psi(param, xs):
  del param
  # log of exact hydrogen wavefunction.
  return -jnp.abs(jnp.linalg.norm(xs))


def h_atom_log_psi_signed(param, xs):
  log_psi = h_atom_log_psi(param, xs)
  return jnp.ones_like(log_psi), log_psi


def kinetic_from_hessian(log_f):

  def kinetic_operator(params, x):
    f = lambda x: jnp.exp(log_f(params, x))
    ys = f(x)
    hess = jax.hessian(f)(x)
    return -0.5 * jnp.trace(hess) / ys

  return kinetic_operator


def kinetic_from_hessian_log(log_f):

  def kinetic_operator(params, x):
    f = lambda x: log_f(params, x)
    grad_f = jax.grad(f)(x)
    hess = jax.hessian(f)(x)
    return -0.5 * (jnp.trace(hess)  + jnp.sum(grad_f**2))

  return kinetic_operator


class HamiltonianTest(parameterized.TestCase):

  def test_local_kinetic_energy(self):

    dummy_params = {}
    xs = np.random.normal(size=(3,))
    expected_kinetic_energy = -(1 - 2 / np.abs(np.linalg.norm(xs))) / 2

    kinetic = hamiltonian.local_kinetic_energy(h_atom_log_psi)
    kinetic_energy = kinetic(dummy_params, xs)
    np.testing.assert_allclose(
        kinetic_energy, expected_kinetic_energy, rtol=1.e-5)

  def test_potential_energy_null(self):

    # with one electron and a nuclear charge of zero, the potential energy is
    # zero.
    xs = np.random.normal(size=(1, 3))
    r_ae = np.linalg.norm(xs, axis=-1)
    r_ee = np.zeros(shape=(1, 1, 1))
    atoms = np.zeros(shape=(1, 3))
    charges = np.zeros(shape=(1,))
    v = hamiltonian.potential_energy(r_ae, r_ee, atoms, charges)
    np.testing.assert_allclose(v, 0.0, rtol=1E-5)

  def test_potential_energy_ee(self):

    xs = np.random.normal(size=(5, 3))
    r_ae = np.linalg.norm(xs, axis=-1)
    r_ee = np.linalg.norm(xs[None, ...] - xs[:, None, :], axis=-1)
    atoms = np.zeros(shape=(1, 3))
    charges = np.zeros(shape=(1,))
    mask = ~np.eye(r_ee.shape[0], dtype=bool)
    expected_v_ee = 0.5 * np.sum(1.0 / r_ee[mask])
    v = hamiltonian.potential_energy(r_ae, r_ee[..., None], atoms, charges)
    np.testing.assert_allclose(v, expected_v_ee, rtol=1E-5)

  def test_potential_energy_he2_ion(self):

    xs = np.random.normal(size=(1, 3))
    atoms = np.array([[
        0,
        0,
        -1,
    ], [0, 0, 1]])
    r_ae = np.linalg.norm(xs - atoms, axis=-1)
    r_ee = np.zeros(shape=(1, 1, 1))
    charges = np.array([2, 2])
    v_ee = -np.sum(charges / r_ae)
    v_ae = np.prod(charges) / np.linalg.norm(np.diff(atoms, axis=0))
    expected_v = v_ee + v_ae
    v = hamiltonian.potential_energy(r_ae[..., None], r_ee, atoms, charges)
    np.testing.assert_allclose(v, expected_v, rtol=1E-5)

  def test_local_energy(self):

    atoms = np.zeros(shape=(1, 3))
    charges = np.ones(shape=(1,))
    dummy_params = {}
    local_energy = hamiltonian.local_energy(
        h_atom_log_psi_signed, atoms, charges, nspins=(1, 0), use_scan=False)

    xs = np.random.normal(size=(100, 3))
    key = jax.random.PRNGKey(4)
    keys = jax.random.split(key, num=xs.shape[0])
    batch_local_energy = jax.vmap(local_energy, in_axes=(None, 0, 0))
    energies = batch_local_energy(dummy_params, keys, xs)

    np.testing.assert_allclose(
        energies, -0.5 * np.ones_like(energies), rtol=1E-5)


class LaplacianTest(parameterized.TestCase):

  def test_laplacian(self):

    xs = np.random.uniform(size=(100, 3))
    dummy_params = {}
    t_l = jax.vmap(
        hamiltonian.local_kinetic_energy(h_atom_log_psi),
        in_axes=(None, 0))(dummy_params, xs)
    hess_t = jax.vmap(
        kinetic_from_hessian(h_atom_log_psi), in_axes=(None, 0))(None, xs)
    np.testing.assert_allclose(t_l, hess_t, rtol=1E-5)

  @parameterized.parameters([True, False])
  def test_fermi_net_laplacian(self, full_det):
    natoms = 2
    np.random.seed(12)
    atoms = np.random.uniform(low=-5.0, high=5.0, size=(natoms, 3))
    nspins = (2, 3)
    charges = list(range(3, 3+natoms*2, 2))
    batch = 4
    cfg = base_config.default()
    cfg.network.full_det = full_det
    cfg.network.detnet.hidden_dims = ((8, 4),)*2
    cfg.network.detnet.determinants = 2
    feature_layer = networks.make_ferminet_features(
        charges,
        cfg.system.electrons,
        cfg.system.ndim,
    )
    network_init, signed_network, _ = networks.make_fermi_net(
        atoms, nspins, charges,
        full_det=full_det,
        feature_layer=feature_layer,
        **cfg.network.detnet
    )
    network = lambda params, x: signed_network(params, x)[1]
    key = jax.random.PRNGKey(47)
    params = network_init(key)
    xs = np.random.normal(scale=5, size=(batch, sum(nspins) * 3))
    t_l_fn = jax.jit(
        jax.vmap(hamiltonian.local_kinetic_energy(network), in_axes=(None, 0)))
    t_l = t_l_fn(params, xs)
    hess_t_fn = jax.jit(
        jax.vmap(kinetic_from_hessian_log(network), in_axes=(None, 0)))
    hess_t = hess_t_fn(params, xs)
    if hess_t.dtype == jnp.float64:
      atol, rtol = 1.e-10, 1.e-10
    else:
      # This needs a low tolerance because on fast math optimization in CPU can
      # substantially affect floating point expressions. See
      # https://github.com/google/jax/issues/6566.
      atol, rtol = 4.e-3, 4.e-3
    np.testing.assert_allclose(t_l, hess_t, atol=atol, rtol=rtol)


if __name__ == '__main__':
  absltest.main()
