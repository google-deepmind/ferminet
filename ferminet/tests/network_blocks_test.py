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
# limitations under the License.

"""Tests for ferminet.network_blocks."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import network_blocks
import numpy as np


class NetworkBlocksTest(parameterized.TestCase):

  @parameterized.parameters([
      {'sizes': [], 'expected_indices': []},
      {'sizes': [3], 'expected_indices': []},
      {'sizes': [3, 0], 'expected_indices': [3]},
      {'sizes': [3, 6], 'expected_indices': [3]},
      {'sizes': [3, 6, 0], 'expected_indices': [3, 9]},
      {'sizes': [2, 0, 6], 'expected_indices': [2, 2]},
  ])
  def test_array_partitions(self, sizes, expected_indices):
    self.assertEqual(network_blocks.array_partitions(sizes), expected_indices)

  @parameterized.parameters(
      {'shape': shape} for shape in [(1, 1, 1), (10, 2, 2), (10, 3, 3)])
  def test_slogdet(self, shape, dtype=np.float32):
    a = np.random.normal(size=shape).astype(dtype)
    s1, ld1 = network_blocks.slogdet(a)
    s2, ld2 = np.linalg.slogdet(a)
    np.testing.assert_allclose(s1, s2, atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(ld1, ld2, atol=1E-5, rtol=1E-5)

  @parameterized.parameters([
      {
          'dims': [4]
      },
      {
          'dims': [4, 3]
      },
      {
          'dims': [4, 3, 1]
      },
  ])
  def test_split_into_blocks(self, dims):

    trailing_dims = [5]
    shape = [sum(dims), sum(dims)] + trailing_dims
    xs = np.random.uniform(size=shape).astype(np.float32)
    blocks = network_blocks.split_into_blocks(xs, dims)

    expected_shapes = [
        list(dims) + trailing_dims for dims in itertools.product(dims, dims)
    ]
    shapes = [list(block.shape) for block in blocks]
    with self.subTest('check shapes'):
      self.assertEqual(shapes, expected_shapes)

    # each group of N blocks was split along axis=1.
    blocks1 = [
        np.concatenate(blocks[i:i + len(dims)], axis=1)
        for i in range(0, len(blocks), len(dims))
    ]
    # groups were split along axis=0.
    reconstructed_xs = np.concatenate(blocks1, axis=0)
    with self.subTest('check can reconstruct original array'):
      np.testing.assert_allclose(xs, reconstructed_xs)


if __name__ == '__main__':
  absltest.main()
