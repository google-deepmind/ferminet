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

"""Tests for ferminet.envelopes."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import envelopes
import jax.numpy as jnp
import numpy as np


def _shape_options(dim2=None):
  shapes = [[(3, 4, 5), (5, 6, 4, 2)], [(3, 4, 1), (1, 6, 4, 2)],
            [(3, 1, 5), (5, 6, 1, 2)], [(3, 1, 1), (1, 6, 1, 2)]]
  for shape1, shape2 in shapes:
    if dim2:
      shape2 = shape2[:dim2]
    yield {
        'testcase_name': f'_shape1={shape1}_shape2={shape2}',
        'shapes': (shape1, shape2),
    }


class ApplyCovarianceTest(parameterized.TestCase):

  @parameterized.named_parameters(_shape_options())
  def test_apply_covariance(self, shapes):
    rng = np.random.RandomState(0).standard_normal
    if jnp.ones(1).dtype == jnp.float64:
      dtype = np.float64
      atol = 0
    else:
      dtype = np.float32
      atol = 1.e-6
    x = rng(shapes[0]).astype(dtype)
    y = rng(shapes[1]).astype(dtype)
    np.testing.assert_allclose(
        envelopes._apply_covariance(x, y),
        jnp.einsum('ijk,kmjn->ijmn', x, y),
        atol=atol,
    )

  @parameterized.named_parameters(_shape_options(dim2=3))
  def test_reduced_apply_covariance(self, shapes):
    rng = np.random.RandomState(0).standard_normal
    if jnp.ones(1).dtype == jnp.float64:
      dtype = np.float64
      atol = 0
    else:
      dtype = np.float32
      atol = 1.e-6
    x = rng(shapes[0]).astype(dtype)
    y = rng(shapes[1]).astype(dtype)
    np.testing.assert_allclose(
        jnp.squeeze(
            envelopes._apply_covariance(x, jnp.expand_dims(y, -1)), axis=-1),
        jnp.einsum('ijk,klj->ijl', x, y),
        atol=atol,
    )


if __name__ == '__main__':
  absltest.main()
