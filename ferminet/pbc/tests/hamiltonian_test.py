# Copyright 2022 DeepMind Technologies Limited.
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
# limitations under the License

"""Tests for ferminet.pbc.hamiltonian."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import networks
from ferminet.pbc import envelopes
from ferminet.pbc import feature_layer as pbc_feature_layer
from ferminet.pbc import hamiltonian
import jax
import jax.numpy as jnp
import numpy as np


class PbcHamiltonianTest(parameterized.TestCase):

  def test_periodicity(self):
    cfg = base_config.default()

    nspins = (6, 5)
    atoms = jnp.asarray([[0., 0., 0.2], [1.2, 1., -0.2], [2.5, -0.8, 0.6]])
    natom = atoms.shape[0]
    charges = jnp.asarray([2, 5, 7])
    spins = np.ones(shape=(1,))
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    xs = jax.random.uniform(subkey, shape=(sum(nspins), 3))

    feature_layer = pbc_feature_layer.make_pbc_feature_layer(
        natom, nspins, ndim=3, lattice=jnp.eye(3), include_r_ae=False
    )

    kpoints = envelopes.make_kpoints(jnp.eye(3), nspins)

    network = networks.make_fermi_net(
        nspins,
        charges,
        envelope=envelopes.make_multiwave_envelope(kpoints),
        feature_layer=feature_layer,
        bias_orbitals=cfg.network.bias_orbitals,
        full_det=cfg.network.full_det,
        **cfg.network.ferminet,
    )

    key, subkey = jax.random.split(key)
    params = network.init(subkey)

    local_energy = hamiltonian.local_energy(
        f=network.apply,
        charges=charges,
        nspins=nspins,
        use_scan=False,
        lattice=jnp.eye(3),
        heg=False,
    )

    data = networks.FermiNetData(
        positions=xs.flatten(), spins=spins, atoms=atoms, charges=charges
    )

    key, subkey = jax.random.split(key)
    e1, _ = local_energy(params, subkey, data)

    # Select random electron coordinate to displace by a random lattice vec
    key, subkey = jax.random.split(key)
    e_idx = jax.random.randint(subkey, (1,), 0, xs.shape[0])
    key, subkey = jax.random.split(key)
    randvec = jax.random.randint(subkey, (3,), 0, 100).astype(jnp.float32)
    xs = xs.at[e_idx].add(randvec)

    data2 = networks.FermiNetData(
        positions=xs.flatten(), spins=spins, atoms=atoms, charges=charges
    )

    key, subkey = jax.random.split(key)
    e2, _ = local_energy(params, subkey, data2)

    atol, rtol = 4.e-3, 4.e-3
    np.testing.assert_allclose(e1, e2, atol=atol, rtol=rtol)


if __name__ == '__main__':
  absltest.main()
