# Copyright 2024 DeepMind Technologies Limited.
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

"""Test experiments from Pfau, Axelrod, Sutterud, von Glehn, Spencer (2024)."""
import itertools
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import chex
from ferminet import base_config
from ferminet import train
from ferminet.configs.excited import atoms
from ferminet.configs.excited import benzene
from ferminet.configs.excited import carbon_dimer
from ferminet.configs.excited import double_excitation
from ferminet.configs.excited import oscillator
from ferminet.configs.excited import presets
from ferminet.configs.excited import twisted_ethylene
import jax

import pyscf

FLAGS = flags.FLAGS
# Default flags are sufficient so mark FLAGS as parsed so we can run the tests
# with py.test, which imports this file rather than runs it.
FLAGS.mark_as_parsed()


def setUpModule():
  # Allow chex_n_cpu_devices to be set via an environment variable as well as
  # --chex_n_cpu_devices to play nicely with pytest.
  fake_devices = os.environ.get('FERMINET_CHEX_N_CPU_DEVICES')
  if fake_devices is not None:
    fake_devices = int(fake_devices)
  try:
    chex.set_n_cpu_devices(n=fake_devices)
  except RuntimeError:
    # jax has already been initialised (e.g. because this is being run with
    # other tests via a test runner such as pytest)
    logging.info('JAX already initialised so cannot set number of CPU devices. '
                 'Using a single device in train_test.')
  jax.clear_caches()


def minimize_config(cfg):
  """Change fields in config to minimal values appropriate for fast testing."""
  cfg.network.ferminet.hidden_dims = ((16, 4),) * 2
  cfg.network.psiformer.heads_dim = 4
  cfg.network.psiformer.mlp_hidden_dims = (16,)
  cfg.network.determinants = 2
  cfg.batch_size = 32
  cfg.pretrain.iterations = 10
  cfg.mcmc.burn_in = 10
  cfg.optim.iterations = 3
  return cfg


class ExcitedStateTest(parameterized.TestCase):

  def setUp(self):
    super(ExcitedStateTest, self).setUp()
    # disable use of temp directory in pyscf.
    # Test calculations are small enough to fit in RAM and we don't need
    # checkpoint files.
    pyscf.lib.param.TMPDIR = None

  @parameterized.parameters(itertools.product(
      ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne'],
      [(5, 'ferminet'), (5, 'psiformer'), (10, 'psiformer')]))
  def test_atoms(self, atom, state_and_network):
    """Test experiments from Fig. 1 of Pfau et al. (2024)."""
    states, network = state_and_network
    cfg = atoms.get_config()
    cfg.system.atom = atom
    cfg.system.states = states
    match network:
      case 'ferminet':
        cfg.update_from_flattened_dict(presets.ferminet)
      case 'psiformer':
        cfg.update_from_flattened_dict(presets.psiformer)
      case _:
        raise ValueError(f'Unknown network type: {network}')
    cfg = minimize_config(cfg)
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)

  @parameterized.parameters(itertools.product(
      ['BH', 'HCl', 'H2O', 'H2S', 'BF', 'CO', 'C2H4',
       'CH2O', 'CH2S', 'HNO', 'HCF', 'H2CSi'],
      ['ferminet', 'psiformer']))
  def test_oscillator(self, system, network):
    """Test experiments from Fig. 2 of Pfau et al. (2024)."""
    cfg = oscillator.get_config()
    cfg.system.molecule_name = system
    if system in ['HNO', 'HCF']:
      cfg.pretrain.excitation_type = 'random'
    match network:
      case 'ferminet':
        cfg.update_from_flattened_dict(presets.ferminet)
      case 'psiformer':
        cfg.update_from_flattened_dict(presets.psiformer)
      case _:
        raise ValueError(f'Unknown network type: {network}')
    cfg = minimize_config(cfg)
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)

  @parameterized.parameters(
      [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5])
  def test_carbon_dimer(self, equilibrium_multiple):
    """Test experiments from Fig. 3 of Pfau et al. (2024)."""
    cfg = carbon_dimer.get_config()
    cfg.system.equilibrium_multiple = equilibrium_multiple
    cfg = minimize_config(cfg)
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)

  @parameterized.parameters(itertools.chain(
      itertools.product(['planar'], [0, 15, 30, 45, 60, 70, 80, 85, 90]),
      itertools.product(['twisted'], [0, 20, 40, 60, 70, 80, 90, 95, 97.5,
                                      100, 102.5, 105, 110, 120]),
  ))
  def test_twisted_ethylene(self, system, angle):
    """Test experiments from Fig. 4 of Pfau et al. (2024)."""
    cfg = twisted_ethylene.get_config()
    cfg.system.molecule_name = system
    match system:
      case 'planar':
        cfg.system.twist.tau = angle
      case 'twisted':
        cfg.system.twist.phi = angle
      case _:
        raise ValueError(f'Unknown system type: {system}')
    cfg = minimize_config(cfg)
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)

  @parameterized.parameters(
      itertools.product(['nitrosomethane',
                         'butadiene',
                         'glyoxal',
                         'tetrazine',
                         'cyclopentadienone'],
                        ['ferminet']))
  def test_double_excitation(self, system, network):
    """Test experiments from Fig. 5 of Pfau et al. (2024)."""
    cfg = double_excitation.get_config()
    cfg.system.molecule_name = system

    match system:
      case 'nitrosomethane':
        cfg.system.states = 6
        cfg.pretrain.excitation_type = 'random'
      case 'butadiene':
        cfg.system.states = 7
        cfg.pretrain.excitation_type = 'random'
      case 'glyoxal':
        cfg.system.states = 7
        cfg.pretrain.excitation_type = 'random'
      case 'tetrazine':
        cfg.system.states = 5
        cfg.optim.spin_energy = 0.5
        cfg.mcmc.blocks = 4
      case 'cyclopentadienone':
        cfg.system.states = 6
        cfg.optim.spin_energy = 1.0
        cfg.mcmc.blocks = 4
      case _:
        raise ValueError(f'Unknown system type: {system}')

    match network:
      case 'ferminet':
        cfg.update_from_flattened_dict(presets.ferminet)
      case'psiformer':
        cfg.update_from_flattened_dict(presets.psiformer)
      case _:
        raise ValueError(f'Unknown network type: {network}')

    cfg = minimize_config(cfg)
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)

  @parameterized.parameters(['ferminet', 'psiformer'])
  def test_benzene(self, network):
    """Test experiments from Fig. 6 of Pfau et al. (2024)."""
    cfg = benzene.get_config()
    match network:
      case 'ferminet':
        cfg.update_from_flattened_dict(presets.ferminet)
      case'psiformer':
        cfg.update_from_flattened_dict(presets.psiformer)
      case _:
        raise ValueError(f'Unknown network type: {network}')
    cfg = minimize_config(cfg)
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)

  @parameterized.parameters(itertools.product(
      ['HNO', 'HCF'],
      ['ferminet', 'psiformer'],
      ['vmc', 'vmc_overlap'],
      ['ordered', 'random']))
  def test_pretrain_and_penalty(self, system, network, objective, excitation):
    """Test experiments from Fig. S3 of Pfau et al. (2024)."""
    cfg = oscillator.get_config()
    cfg.system.molecule_name = system
    cfg.pretrain.excitation_type = excitation
    cfg.optim.objective = objective
    match network:
      case 'ferminet':
        cfg.update_from_flattened_dict(presets.ferminet)
      case 'psiformer':
        cfg.update_from_flattened_dict(presets.psiformer)
      case _:
        raise ValueError(f'Unknown network type: {network}')
    cfg = minimize_config(cfg)
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)


if __name__ == '__main__':
  absltest.main()
