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

"""Tests for ferminet.pbc.feature_layer."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import networks
from ferminet.pbc import feature_layer as pbc_feature_layer
import jax
import jax.numpy as jnp
import numpy as np


class FeatureLayerTest(parameterized.TestCase):

  @parameterized.parameters([True, False])
  def test_shape(self, heg):
    """Asserts that output shape of apply matches what is expected by init."""
    nspins = (6, 5)
    atoms = jnp.asarray([[0., 0., 0.2], [1.2, 1., -0.2], [2.5, -0.8, 0.6]])
    natom = atoms.shape[0]
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    xs = jax.random.uniform(subkey, shape=(sum(nspins), 3))

    feature_layer = pbc_feature_layer.make_pbc_feature_layer(
        natom, nspins, 3, lattice=jnp.eye(3), include_r_ae=heg
    )

    dims, params = feature_layer.init()
    ae, ee, r_ae, r_ee = networks.construct_input_features(xs, atoms)

    ae_features, ee_features = feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params)

    assert dims[0] == ae_features.shape[-1]
    assert dims[1] == ee_features.shape[-1]

  def test_periodicity(self):
    nspins = (6, 5)
    atoms = jnp.asarray([[0., 0., 0.2], [1.2, 1., -0.2], [2.5, -0.8, 0.6]])
    natom = atoms.shape[0]
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    xs = jax.random.uniform(subkey, shape=(sum(nspins), 3))

    feature_layer = pbc_feature_layer.make_pbc_feature_layer(
        natom, nspins, 3, lattice=jnp.eye(3), include_r_ae=False
    )

    _, params = feature_layer.init()
    ae, ee, r_ae, r_ee = networks.construct_input_features(xs, atoms)

    ae_features_1, ee_features_1 = feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params)

    # Select random electron coordinate to displace by a random lattice vector
    key, subkey = jax.random.split(key)
    e_idx = jax.random.randint(subkey, (1,), 0, xs.shape[0])
    key, subkey = jax.random.split(key)
    randvec = jax.random.randint(subkey, (3,), 0, 100).astype(jnp.float32)
    xs = xs.at[e_idx].add(randvec)

    ae, ee, r_ae, r_ee = networks.construct_input_features(xs, atoms)

    ae_features_2, ee_features_2 = feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params)

    atol, rtol = 4.e-3, 4.e-3
    np.testing.assert_allclose(
        ae_features_1, ae_features_2, atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        ee_features_1, ee_features_2, atol=atol, rtol=rtol)


if __name__ == '__main__':
  absltest.main()
