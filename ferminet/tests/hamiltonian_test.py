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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import hamiltonian
from ferminet import networks
import jax
import jax.numpy as jnp
import numpy as np


def h_atom_log_psi(param, xs, spins, atoms=None, charges=None):
  del param, spins, atoms, charges
  # log of exact hydrogen wavefunction.
  return -jnp.abs(jnp.linalg.norm(xs))


def h_atom_log_psi_signed(param, xs, spins, atoms=None, charges=None):
  log_psi = h_atom_log_psi(param, xs, spins, atoms, charges)
  return jnp.ones_like(log_psi), log_psi


def kinetic_from_hessian(log_f):

  def kinetic_operator(params, pos, spins, atoms, charges):
    f = lambda x: jnp.exp(log_f(params, x, spins, atoms, charges))
    ys = f(pos)
    hess = jax.hessian(f)(pos)
    return -0.5 * jnp.trace(hess) / ys

  return kinetic_operator


def kinetic_from_hessian_log(log_f):

  def kinetic_operator(params, pos, spins, atoms, charges):
    f = lambda x: log_f(params, x, spins, atoms, charges)
    grad_f = jax.grad(f)(pos)
    hess = jax.hessian(f)(pos)
    return -0.5 * (jnp.trace(hess)  + jnp.sum(grad_f**2))

  return kinetic_operator


class HamiltonianTest(parameterized.TestCase):

  @parameterized.parameters(['default', 'folx'])
  def test_local_kinetic_energy(self, laplacian):

    dummy_params = {}
    xs = np.random.normal(size=(3,))
    spins = np.ones(shape=(1,))
    atoms = np.random.normal(size=(1, 3))
    charges = 2 * np.ones(shape=(1,))
    expected_kinetic_energy = -(1 - 2 / np.abs(np.linalg.norm(xs))) / 2

    kinetic = hamiltonian.local_kinetic_energy(h_atom_log_psi_signed,
                                               laplacian_method=laplacian)
    kinetic_energy = kinetic(
        dummy_params,
        networks.FermiNetData(
            positions=xs, spins=spins, atoms=atoms, charges=charges
        ),
    )
    np.testing.assert_allclose(
        kinetic_energy, expected_kinetic_energy, rtol=1.e-5)

  def test_potential_energy_null(self):

    # with one electron and a nuclear charge of zero, the potential energy is
    # zero.
    xs = np.random.normal(size=(1, 3))
    r_ae = jnp.linalg.norm(xs, axis=-1)
    r_ee = jnp.zeros(shape=(1, 1, 1))
    atoms = jnp.zeros(shape=(1, 3))
    charges = jnp.zeros(shape=(1,))
    v = hamiltonian.potential_energy(r_ae, r_ee, atoms, charges)
    np.testing.assert_allclose(v, 0.0, rtol=1E-5)

  def test_potential_energy_ee(self):

    xs = np.random.normal(size=(5, 3))
    r_ae = jnp.linalg.norm(xs, axis=-1)
    r_ee = jnp.linalg.norm(xs[None, ...] - xs[:, None, :], axis=-1)
    atoms = jnp.zeros(shape=(1, 3))
    charges = jnp.zeros(shape=(1,))
    mask = ~jnp.eye(r_ee.shape[0], dtype=bool)
    expected_v_ee = 0.5 * np.sum(1.0 / r_ee[mask])
    v = hamiltonian.potential_energy(r_ae, r_ee[..., None], atoms, charges)
    np.testing.assert_allclose(v, expected_v_ee, rtol=1E-5)

  def test_potential_energy_he2_ion(self):

    xs = np.random.normal(size=(1, 3))
    atoms = jnp.array([[0, 0, -1], [0, 0, 1]])
    r_ae = jnp.linalg.norm(xs - atoms, axis=-1)
    r_ee = jnp.zeros(shape=(1, 1, 1))
    charges = jnp.array([2, 2])
    v_ee = -jnp.sum(charges / r_ae)
    v_ae = jnp.prod(charges) / jnp.linalg.norm(jnp.diff(atoms, axis=0))
    expected_v = v_ee + v_ae
    v = hamiltonian.potential_energy(r_ae[..., None], r_ee, atoms, charges)
    np.testing.assert_allclose(v, expected_v, rtol=1E-5)

  def test_local_energy(self):

    spins = np.ones(shape=(1,))
    atoms = np.zeros(shape=(1, 3))
    charges = np.ones(shape=(1,))
    dummy_params = {}
    local_energy = hamiltonian.local_energy(
        h_atom_log_psi_signed, charges, nspins=(1, 0), use_scan=False
    )

    xs = np.random.normal(size=(100, 3))
    key = jax.random.PRNGKey(4)
    keys = jax.random.split(key, num=xs.shape[0])
    batch_local_energy = jax.vmap(
        local_energy,
        in_axes=(
            None,
            0,
            networks.FermiNetData(
                positions=0, spins=None, atoms=None, charges=None
            ),
        ),
    )
    energies, _ = batch_local_energy(
        dummy_params,
        keys,
        networks.FermiNetData(
            positions=xs, spins=spins, atoms=atoms, charges=charges
        ),
    )

    np.testing.assert_allclose(
        energies, -0.5 * np.ones_like(energies), rtol=1E-5)


class LaplacianTest(parameterized.TestCase):

  @parameterized.parameters(['default', 'folx'])
  def test_laplacian(self, laplacian):

    xs = np.random.uniform(size=(100, 3))
    spins = np.ones(shape=(1,))
    atoms = np.random.normal(size=(1, 3))
    charges = 3 * np.ones(shape=(1,))
    data = networks.FermiNetData(
        positions=xs, spins=spins, atoms=atoms, charges=charges
    )
    dummy_params = {}
    t_l_fn = jax.vmap(
        hamiltonian.local_kinetic_energy(h_atom_log_psi_signed,
                                         laplacian_method=laplacian),
        in_axes=(
            None,
            networks.FermiNetData(
                positions=0, spins=None, atoms=None, charges=None
            ),
        ),
    )
    t_l = t_l_fn(dummy_params, data)
    hess_t = jax.vmap(
        kinetic_from_hessian(h_atom_log_psi),
        in_axes=(None, 0, None, None, None),
    )(dummy_params, xs, spins, atoms, charges)
    np.testing.assert_allclose(t_l, hess_t, rtol=1E-5)

  @parameterized.parameters(
      itertools.product([True, False], ['default', 'folx'])
  )
  def test_fermi_net_laplacian(self, full_det, laplacian):
    natoms = 2
    np.random.seed(12)
    atoms = np.random.uniform(low=-5.0, high=5.0, size=(natoms, 3))
    nspins = (2, 3)
    charges = 2 * np.ones(shape=(natoms,))
    batch = 4
    cfg = base_config.default()
    cfg.network.full_det = full_det
    cfg.network.ferminet.hidden_dims = ((8, 4),) * 2
    cfg.network.determinants = 2
    feature_layer = networks.make_ferminet_features(
        natoms,
        cfg.system.electrons,
        cfg.system.ndim,
    )
    network = networks.make_fermi_net(
        nspins,
        charges,
        full_det=full_det,
        feature_layer=feature_layer,
        **cfg.network.ferminet
    )
    log_network = lambda *args, **kwargs: network.apply(*args, **kwargs)[1]
    key = jax.random.PRNGKey(47)
    params = network.init(key)
    xs = np.random.normal(scale=5, size=(batch, sum(nspins) * 3))
    spins = np.sign(np.random.normal(scale=1, size=(batch, sum(nspins))))
    t_l_fn = jax.jit(
        jax.vmap(
            hamiltonian.local_kinetic_energy(network.apply,
                                             laplacian_method=laplacian),
            in_axes=(
                None,
                networks.FermiNetData(
                    positions=0, spins=0, atoms=None, charges=None
                ),
            ),
        )
    )
    t_l = t_l_fn(
        params,
        networks.FermiNetData(
            positions=xs, spins=spins, atoms=atoms, charges=charges
        ),
    )
    hess_t_fn = jax.jit(
        jax.vmap(
            kinetic_from_hessian_log(log_network),
            in_axes=(None, 0, 0, None, None),
        )
    )
    hess_t = hess_t_fn(params, xs, spins, atoms, charges)
    if hess_t.dtype == jnp.float64:
      atol, rtol = 1.e-10, 1.e-10
    else:
      # This needs a low tolerance because on fast math optimization in CPU can
      # substantially affect floating point expressions. See
      # https://github.com/jax-ml/jax/issues/6566.
      atol, rtol = 4.e-3, 4.e-3
    np.testing.assert_allclose(t_l, hess_t, atol=atol, rtol=rtol)


if __name__ == '__main__':
  absltest.main()
