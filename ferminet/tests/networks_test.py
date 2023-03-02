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

"""Tests for ferminet.networks."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from ferminet import envelopes
from ferminet import networks
import jax
from jax import random
import jax.numpy as jnp
import numpy as np


def rand_default():
  randn = np.random.RandomState(0).randn
  def generator(shape, dtype):
    return randn(*shape).astype(dtype)
  return generator


def _antisymmtry_options():
  for envelope in envelopes.EnvelopeLabel:
    yield {
        'testcase_name': f'_envelope={envelope}',
        'envelope_label': envelope,
        'dtype': np.float32,
    }


def _network_options():
  """Yields the set of all combinations of options to pass into test_fermi_net.

  Example output:
  {
    'vmap': True,
    'envelope': envelopes.EnvelopeLabel.ISOTROPIC,
    'bias_orbitals': False,
    'full_det': True,
    'use_last_layer': False,
    'hidden_dims': ((32, 8), (32, 8)),
  }
  """
  # Key for each option and corresponding values to test.
  all_options = {
      'vmap': [True, False],
      'envelope_label': list(envelopes.EnvelopeLabel),
      'bias_orbitals': [True, False],
      'full_det': [True, False],
      'use_last_layer': [True, False],
      'hidden_dims': [((32, 8), (32, 8))],
  }
  # Create the product of all options.
  for options in itertools.product(*all_options.values()):
    # Yield dict of the current combination of options.
    yield dict(zip(all_options.keys(), options))


class NetworksTest(parameterized.TestCase):

  @parameterized.named_parameters(_antisymmtry_options())
  def test_antisymmetry(self, envelope_label, dtype):
    """Check that the Fermi Net is symmetric."""
    del dtype  # unused

    key = random.PRNGKey(42)

    key, *subkeys = random.split(key, num=3)
    natoms = 4
    atoms = random.normal(subkeys[0], shape=(natoms, 3))
    charges = random.normal(subkeys[1], shape=(natoms,))
    nspins = (3, 4)

    key, subkey = random.split(key)
    pos1 = random.normal(subkey, shape=(sum(nspins) * 3,))
    pos2 = jnp.concatenate((pos1[3:6], pos1[:3], pos1[6:]))
    pos3 = jnp.concatenate((pos1[:9], pos1[12:15], pos1[9:12], pos1[15:]))

    key, subkey = random.split(key)
    spins1 = jax.random.uniform(subkey, shape=(sum(nspins),))
    spins2 = jnp.concatenate((spins1[1:2], spins1[:1], spins1[2:]))
    spins3 = jnp.concatenate((spins1[:3], spins1[4:5], spins1[3:4], spins1[5:]))

    feature_layer = networks.make_ferminet_features(natoms, nspins, ndim=3)

    kwargs = {}
    network = networks.make_fermi_net(
        nspins=nspins,
        charges=charges,
        hidden_dims=((16, 16), (16, 16)),
        envelope=envelopes.get_envelope(envelope_label, **kwargs),
        feature_layer=feature_layer,
    )

    key, subkey = random.split(key)
    params = network.init(subkey)

    # Randomize parameters of envelope
    if isinstance(params['envelope'], list):
      for i in range(len(params['envelope'])):
        if params['envelope'][i]:
          key, *subkeys = random.split(key, num=3)
          params['envelope'][i]['sigma'] = random.normal(
              subkeys[0], params['envelope'][i]['sigma'].shape)
          params['envelope'][i]['pi'] = random.normal(
              subkeys[1], params['envelope'][i]['pi'].shape)
    else:
      assert isinstance(params['envelope'], dict)
      key, *subkeys = random.split(key, num=3)
      params['envelope']['sigma'] = random.normal(
          subkeys[0], params['envelope']['sigma'].shape)
      params['envelope']['pi'] = random.normal(
          subkeys[1], params['envelope']['pi'].shape
      )

    out1 = network.apply(params, pos1, spins1, atoms, charges)

    out2 = network.apply(params, pos2, spins2, atoms, charges)
    np.testing.assert_allclose(out1[1], out2[1], atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(out1[0], -1*out2[0], atol=1E-5, rtol=1E-5)

    out3 = network.apply(params, pos3, spins3, atoms, charges)
    np.testing.assert_allclose(out1[1], out3[1], atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(out1[0], -1*out3[0], atol=1E-5, rtol=1E-5)

  def test_create_input_features(self):
    dtype = np.float32
    ndim = 3
    nelec = 6
    xs = np.random.normal(scale=3, size=(nelec, ndim)).astype(dtype)
    atoms = jnp.array([[0.2, 0.5, 0.3], [1.2, 0.3, 0.7]])
    input_features = networks.construct_input_features(xs, atoms)
    d_input_features = jax.jacfwd(networks.construct_input_features)(
        xs, atoms, ndim=3)
    r_ee = input_features[-1][:, :, 0]
    d_r_ee = d_input_features[-1][:, :, 0]
    # The gradient of |r_i - r_j| wrt r_k should only be non-zero for k = i or
    # k = j and the i = j term should be explicitly masked out.
    mask = np.fromfunction(
        lambda i, j, k: np.logical_and(np.logical_or(i == k, j == k), i != j),
        d_r_ee.shape[:-1],
    )
    d_r_ee_non_zeros = d_r_ee[mask]
    d_r_ee_zeros = d_r_ee[~mask]
    with self.subTest('check forward pass'):
      chex.assert_tree_all_finite(input_features)
      # |r_i - r_j| should be zero.
      np.testing.assert_allclose(np.diag(r_ee), np.zeros(6), atol=1E-5)
    with self.subTest('check backwards pass'):
      # Most importantly, check the gradient of the electron-electron distances,
      # |x_i - x_j|, is masked out for i==j.
      chex.assert_tree_all_finite(d_input_features)
      # We mask out the |r_i-r_j| terms for i == j. Check these are zero.
      np.testing.assert_allclose(
          d_r_ee_zeros, np.zeros_like(d_r_ee_zeros), atol=1E-5, rtol=1E-5)
      self.assertTrue(np.all(np.abs(d_r_ee_non_zeros) > 0.0))

  @parameterized.parameters(None, 4)
  def test_construct_symmetric_features(self, naux_features):
    dtype = np.float32
    hidden_units_one = 8  # 128
    hidden_units_two = 4  # 32
    nspins = (6, 5)
    h_one = np.random.uniform(
        low=-5, high=5, size=(sum(nspins), hidden_units_one)).astype(dtype)
    h_two = np.random.uniform(
        low=-5,
        high=5,
        size=(sum(nspins), sum(nspins), hidden_units_two)).astype(dtype)
    if naux_features:
      h_aux = np.random.uniform(size=(sum(nspins), naux_features)).astype(dtype)
    else:
      h_aux = None
    h_two = h_two + np.transpose(h_two, axes=(1, 0, 2))
    features = networks.construct_symmetric_features(
        h_one, h_two, nspins, h_aux=h_aux
    )
    # Swap electrons
    swaps = np.arange(sum(nspins))
    np.random.shuffle(swaps[:nspins[0]])
    np.random.shuffle(swaps[nspins[0]:])
    inverse_swaps = [0] * len(swaps)
    for i, j in enumerate(swaps):
      inverse_swaps[j] = i
    inverse_swaps = np.asarray(inverse_swaps)
    h_aux_swap = h_aux if h_aux is None else h_aux[swaps]
    features_swap = networks.construct_symmetric_features(
        h_one[swaps], h_two[swaps][:, swaps], nspins, h_aux=h_aux_swap
    )
    np.testing.assert_allclose(
        features, features_swap[inverse_swaps], atol=1E-5, rtol=1E-5)

  @parameterized.parameters(_network_options())
  def test_fermi_net(self, vmap, **network_options):
    # Warning: this only tests we can build and run the network. It does not
    # test correctness of output nor test changing network width or depth.
    nspins = (6, 5)
    natoms = 3
    atoms = jnp.asarray([[0., 0., 0.2], [1.2, 1., -0.2], [2.5, -0.8, 0.6]])
    charges = jnp.asarray([2, 5, 7])
    key = jax.random.PRNGKey(42)
    feature_layer = networks.make_ferminet_features(natoms, nspins, ndim=3)
    kwargs = {}
    network_options['envelope'] = envelopes.get_envelope(
        network_options['envelope_label'], **kwargs)
    del network_options['envelope_label']

    envelope = network_options['envelope']
    if (
        envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL
        and network_options['bias_orbitals']
    ):
      with self.assertRaises(ValueError):
        networks.make_fermi_net(
            nspins, charges, feature_layer=feature_layer, **network_options
        )
    else:
      network = networks.make_fermi_net(
          nspins, charges, feature_layer=feature_layer, **network_options
      )

      key, *subkeys = jax.random.split(key, num=3)
      if vmap:
        batch = 10
        xs = jax.random.uniform(subkeys[0], shape=(batch, sum(nspins), 3))
        spins = jax.random.uniform(subkeys[1], shape=(batch, sum(nspins)))
        fermi_net = jax.vmap(network.apply, in_axes=(None, 0, 0, None, None))
        expected_shape = (batch,)
      else:
        xs = jax.random.uniform(subkeys[0], shape=(sum(nspins), 3))
        spins = jax.random.uniform(subkeys[1], shape=(sum(nspins),))
        fermi_net = network.apply
        expected_shape = ()

      key, subkey = jax.random.split(key)
      params = network.init(subkey)
      sign_psi, log_psi = fermi_net(params, xs, spins, atoms, charges)
      self.assertSequenceEqual(sign_psi.shape, expected_shape)
      self.assertSequenceEqual(log_psi.shape, expected_shape)

  @parameterized.parameters(
      *(itertools.product([(1, 0), (2, 0), (0, 1)], [True, False])))
  def test_spin_polarised_fermi_net(self, nspins, full_det):
    natoms = 1
    atoms = jnp.zeros(shape=(1, 3))
    charges = jnp.ones(shape=1)
    key = jax.random.PRNGKey(42)
    feature_layer = networks.make_ferminet_features(natoms, nspins, ndim=3)
    network = networks.make_fermi_net(
        nspins, charges, feature_layer=feature_layer, full_det=full_det
    )
    key, *subkeys = jax.random.split(key, num=4)
    params = network.init(subkeys[0])
    xs = jax.random.uniform(subkeys[1], shape=(sum(nspins) * 3,))
    spins = jax.random.uniform(subkeys[2], shape=(sum(nspins),))
    # Test fermi_net runs without raising exceptions for spin-polarised systems.
    sign_out, log_out = network.apply(params, xs, spins, atoms, charges)
    self.assertEqual(sign_out.size, 1)
    self.assertEqual(log_out.size, 1)


if __name__ == '__main__':
  absltest.main()
