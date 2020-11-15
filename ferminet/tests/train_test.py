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

"""Tests for ferminet.jax.train."""
import itertools

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import train
from ferminet.configs import atom
from ferminet.configs import diatomic
from jax import test_util as jtu
import pyscf

OPTIMIZERS = ['adam']

FLAGS = flags.FLAGS
# Default flags are sufficient so mark FLAGS as parsed so we can run the tests
# with py.test, which imports this file rather than runs it.
FLAGS.mark_as_parsed()


class QmcTest(jtu.JaxTestCase):

  def setUp(self):
    super(QmcTest, self).setUp()
    # disable use of temp directory in pyscf.
    # Test calculations are small enough to fit in RAM and we don't need
    # checkpoint files.
    pyscf.lib.param.TMPDIR = None

  @parameterized.parameters(
      (system, optimizer)
      for system, optimizer in itertools.product(['Li', 'LiH'], OPTIMIZERS))
  def test_training_step(self, system, optimizer):
    if system == 'Li':
      cfg = atom.get_config()
      cfg.system.atom = system
    else:
      cfg = diatomic.get_config()
      cfg.system.molecule_name = system
    cfg.network.detnet.hidden_dims = ((16, 4),) * 2
    cfg.network.detnet.determinants = 2
    cfg.batch_size = 32
    cfg.pretrain.iterations = 10
    cfg.mcmc.burn_in = 10
    cfg.optim.iterations = 3
    cfg.debug.check_nan = True
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)


if __name__ == '__main__':
  absltest.main()
