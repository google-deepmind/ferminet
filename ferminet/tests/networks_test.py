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

"""Tests for ferminet.tests.networks."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import chex
from ferminet import networks
import jax
from jax import random
from jax import test_util as jtu
import jax.numpy as jnp
import numpy as np


def rand_default():
  randn = np.random.RandomState(0).randn
  def generator(shape, dtype):
    return randn(*shape).astype(dtype)
  return generator


def _network_options():
  """Yields the set of all combinations of options to pass into test_fermi_net.

  Example output:
  {
    'vmap': True,
    'envelope_type': 'isotropic',
    'bias_orbitals': False,
    'full_det': True,
    'use_last_layer': False,
    'hidden_dims': ((32, 8), (32, 8)),
  }
  """
  # Key for each option and corresponding values to test.
  all_options = {
      'vmap': [True, False],
      'envelope_type': [
          'isotropic', 'diagonal', 'full', 'sto', 'sto-poly', 'output'
      ],
      'bias_orbitals': [True, False],
      'full_det': [True, False],
      'use_last_layer': [True, False],
      'hidden_dims': [((32, 8), (32, 8))],
  }
  # Create the product of all options.
  for options in itertools.product(*all_options.values()):
    # Yield dict of the current combination of options.
    yield dict(zip(all_options.keys(), options))


class NetworksTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {'testcase_name':  # pylint: disable=g-complex-comprehension
       '_envelope={}'.format(envelope),
       'envelope': envelope, 'dtype': dtype}
      for envelope in ['isotropic', 'diagonal', 'full', 'output', 'exact_cusp']
      for dtype in [np.float32]))
  def test_antisymmetry(self, envelope, dtype):
    """Check that the Fermi Net is symmetric."""
    key = random.PRNGKey(42)

    key, *subkeys = random.split(key, num=3)
    atoms = random.normal(subkeys[0], shape=(4, 3))
    charges = random.normal(subkeys[1], shape=(4,))
    spins = (3, 4)

    key, subkey = random.split(key)
    data1 = random.normal(subkey, shape=(21,))
    data2 = jnp.concatenate((data1[3:6], data1[:3], data1[6:]))
    data3 = jnp.concatenate((data1[:9], data1[12:15], data1[9:12], data1[15:]))
    key, subkey = random.split(key)
    params = networks.init_fermi_net_params(
        subkey,
        atoms=atoms,
        spins=spins,
        hidden_dims=((16, 16), (16, 16)),
        envelope_type=envelope)

    # Randomize parameters of envelope
    if isinstance(params['envelope'], list):
      for i in range(len(params['envelope'])):
        key, *subkeys = random.split(key, num=3)
        params['envelope'][i]['sigma'] = random.normal(
            subkeys[0], params['envelope'][i]['sigma'].shape)
        params['envelope'][i]['pi'] = random.normal(
            subkeys[1], params['envelope'][i]['pi'].shape)
    else:
      key, *subkeys = random.split(key, num=3)
      params['envelope']['sigma'] = random.normal(
          subkeys[0], params['envelope']['sigma'].shape)
      params['envelope']['pi'] = random.normal(
          subkeys[1], params['envelope']['pi'].shape)

    out1 = networks.fermi_net(params, data1, atoms, spins, charges,
                              envelope_type=envelope)

    out2 = networks.fermi_net(params, data2, atoms, spins, charges,
                              envelope_type=envelope)
    self.assertAllClose(out1[1], out2[1], check_dtypes=False)
    self.assertAllClose(out1[0], -1*out2[0], check_dtypes=False)

    out3 = networks.fermi_net(params, data3, atoms, spins, charges,
                              envelope_type=envelope)
    self.assertAllClose(out1[1], out3[1], check_dtypes=False)
    self.assertAllClose(out1[0], -1*out3[0], check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {'testcase_name':  # pylint: disable=g-complex-comprehension
       '_shape={}'.format(jtu.format_shape_dtype_string(shape, dtype)),
       'shape': shape, 'dtype': dtype, 'rng': rng}
      for shape in [(1, 1, 1), (10, 2, 2), (10, 3, 3)]
      for dtype in [np.float32]
      for rng in [rand_default()]))
  def test_slogdet(self, shape, dtype, rng):
    a = rng(shape, dtype)
    s1, ld1 = networks.slogdet(a)
    s2, ld2 = np.linalg.slogdet(a)
    self.assertAllClose(s1, s2, check_dtypes=False)
    self.assertAllClose(ld1, ld2, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {'testcase_name':  # pylint: disable=g-complex-comprehension
       '_shape1={}_shape2={}'.format(shapes[0], shapes[1]), 'shapes': shapes}
      for shapes in [[(3, 4, 5), (5, 6, 4, 2)],
                     [(3, 4, 1), (1, 6, 4, 2)],
                     [(3, 1, 5), (5, 6, 1, 2)],
                     [(3, 1, 1), (1, 6, 1, 2)]]))
  def test_apply_covariance(self, shapes):
    rng = rand_default()
    dtype = np.float32
    x = rng(shapes[0], dtype)
    y = rng(shapes[1], dtype)
    self.assertAllClose(networks.apply_covariance(x, y),
                        jnp.einsum('ijk,kmjn->ijmn', x, y),
                        check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {'testcase_name':  # pylint: disable=g-complex-comprehension
       '_shape1={}_shape2={}'.format(shapes[0], shapes[1]), 'shapes': shapes}
      for shapes in [[(3, 4, 5), (5, 6, 4)],
                     [(3, 4, 1), (1, 6, 4)],
                     [(3, 1, 5), (5, 6, 1)],
                     [(3, 1, 1), (1, 6, 1)]]))
  def test_reduced_apply_covariance(self, shapes):
    rng = rand_default()
    dtype = np.float32
    x = rng(shapes[0], dtype)
    y = rng(shapes[1], dtype)
    self.assertAllClose(
        jnp.squeeze(networks.apply_covariance(x, jnp.expand_dims(y, -1)),
                    axis=-1),
        jnp.einsum('ijk,klj->ijl', x, y), check_dtypes=False)

  def test_create_input_features(self):
    dtype = np.float32
    ndim = 3
    nelec = 6
    xs = np.random.normal(scale=3, size=(nelec, ndim)).astype(dtype)
    atoms = np.array([[0.2, 0.5, 0.3], [1.2, 0.3, 0.7]])
    input_features = networks.construct_input_features(xs, atoms)
    d_input_features = jax.jacfwd(networks.construct_input_features)(
        xs, atoms, ndim=3)
    r_ee = input_features[-1][:, :, 0]
    d_r_ee = d_input_features[-1][:, :, 0]
    # The gradient of |r_i - r_j| wrt r_k should only be non-zero for k = i or
    # k = j and the i = j term should be explicitly masked out.
    mask = np.fromfunction(
        lambda i, j, k: np.logical_and(np.logical_or(i == k, j == k), i != j),
        d_r_ee.shape[:-1])
    d_r_ee_non_zeros = d_r_ee[mask]
    d_r_ee_zeros = d_r_ee[~mask]
    with self.subTest('check forward pass'):
      chex.assert_tree_all_finite(input_features)
      # |r_i - r_j| should be zero.
      self.assertAllClose(np.diag(r_ee), np.zeros(6))
    with self.subTest('check backwards pass'):
      # Most importantly, check the gradient of the electron-electron distances,
      # |x_i - x_j|, is masked out for i==j.
      chex.assert_tree_all_finite(d_input_features)
      # We mask out the |r_i-r_j| terms for i == j. Check these are zero.
      self.assertAllClose(d_r_ee_zeros, np.zeros_like(d_r_ee_zeros))
      self.assertTrue(np.all(np.abs(d_r_ee_non_zeros) > 0.0))

  def test_construct_symmetric_features(self):
    dtype = np.float32
    hidden_units_one = 8  # 128
    hidden_units_two = 4  # 32
    spins = (6, 5)
    h_one = np.random.uniform(
        low=-5, high=5, size=(sum(spins), hidden_units_one)).astype(dtype)
    h_two = np.random.uniform(
        low=-5,
        high=5,
        size=(sum(spins), sum(spins), hidden_units_two)).astype(dtype)
    h_two = h_two + np.transpose(h_two, axes=(1, 0, 2))
    features = networks.construct_symmetric_features(h_one, h_two, spins)
    # Swap electrons
    swaps = np.arange(sum(spins))
    np.random.shuffle(swaps[:spins[0]])
    np.random.shuffle(swaps[spins[0]:])
    inverse_swaps = [0] * len(swaps)
    for i, j in enumerate(swaps):
      inverse_swaps[j] = i
    inverse_swaps = np.asarray(inverse_swaps)
    features_swap = networks.construct_symmetric_features(
        h_one[swaps], h_two[swaps][:, swaps], spins)
    self.assertAllClose(features, features_swap[inverse_swaps])

  @parameterized.parameters(
      _network_options()
  )
  def test_fermi_net(self, vmap, **network_options):
    # Warning: this only tests we can build and run the network. It does not
    # test correctness of output nor test changing network width or depth.
    spins = (6, 5)
    atoms = jnp.asarray([[0., 0., 0.2], [1.2, 1., -0.2], [2.5, -0.8, 0.6]])
    charges = jnp.asarray([2, 5, 7])
    key = jax.random.PRNGKey(42)

    init, fermi_net = networks.make_fermi_net(atoms, spins, charges,
                                              **network_options)

    key, subkey = jax.random.split(key)
    if vmap:
      batch = 10
      xs = jax.random.uniform(subkey, shape=(batch, sum(spins), 3))
      fermi_net = jax.vmap(fermi_net, in_axes=(None, 0))
      expected_shape = (batch,)
    else:
      xs = jax.random.uniform(subkey, shape=(sum(spins), 3))
      expected_shape = ()

    key, subkey = jax.random.split(key)
    if (network_options['envelope_type'] in ('sto', 'sto-poly') and
        network_options['bias_orbitals']):
      with self.assertRaises(ValueError):
        init(subkey)
    else:
      params = init(subkey)
      result = fermi_net(params, xs)
      self.assertSequenceEqual(result.shape, expected_shape)

  @parameterized.parameters((spins, full_det)
                            for spins, full_det in itertools.product([(
                                1, 0), (2, 0), (0, 1)], [True, False]))
  def test_spin_polarised_fermi_net(self, spins, full_det):
    atoms = jnp.zeros(shape=(1, 3))
    charges = jnp.ones(shape=1)
    key = jax.random.PRNGKey(42)
    init, fermi_net = networks.make_fermi_net(atoms, spins, charges)
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    params = init(subkey1)
    xs = jax.random.uniform(subkey2, shape=(sum(spins) * 3,))
    # Test fermi_net runs without raising exceptions for spin-polarised systems.
    fermi_net(params, xs)


if __name__ == '__main__':
  absltest.main()
