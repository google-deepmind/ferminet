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

"""Tests for ferminet.train."""
import itertools
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import chex
from ferminet import base_config
from ferminet import train
from ferminet.configs import atom
from ferminet.configs import diatomic
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


def _config_params():
  for system, optimizer, complex_, states in itertools.product(
      ('Li', 'LiH'), ('kfac', 'adam'), (True, False), (0, 2)):
    if states == 0 or not complex_:
      yield {'system': system,
             'optimizer': optimizer,
             'complex_': complex_,
             'states': states,
             'laplacian': 'default',
             'scf_fraction': 0.0}
  for states, scf_fraction in itertools.product((0, 2), (0.0, 0.5, 1.0)):
    yield {'system': 'LiH',
           'optimizer': 'kfac',
           'complex_': False,
           'states': states,
           'laplacian': 'default',
           'scf_fraction': scf_fraction}
  for optimizer in ('kfac', 'adam', 'lamb', 'none'):
    yield {
        'system': 'H' if optimizer in ('kfac', 'adam') else 'Li',
        'optimizer': optimizer,
        'complex_': False,
        'states': 0,
        'laplacian': 'default',
        'scf_fraction': 0.0
    }
  for states, laplacian, complex_ in itertools.product(
      (0, 2), ('default', 'folx'), (True, False)):
    yield {
        'system': 'Li',
        'optimizer': 'kfac',
        'complex_': complex_,
        'states': states,
        'laplacian': laplacian,
        'scf_fraction': 0.0
    }


class QmcTest(parameterized.TestCase):

  def setUp(self):
    super(QmcTest, self).setUp()
    # disable use of temp directory in pyscf.
    # Test calculations are small enough to fit in RAM and we don't need
    # checkpoint files.
    pyscf.lib.param.TMPDIR = None

  @parameterized.parameters(_config_params())
  def test_training_step(
      self, system, optimizer, complex_, states, laplacian, scf_fraction):
    if system in ('H', 'Li'):
      cfg = atom.get_config()
      cfg.system.atom = system
    else:
      cfg = diatomic.get_config()
      cfg.system.molecule_name = system
    cfg.network.ferminet.hidden_dims = ((16, 4),) * 2
    cfg.network.determinants = 2
    cfg.network.complex = complex_
    cfg.batch_size = 32
    cfg.system.states = states
    cfg.pretrain.iterations = 10
    cfg.pretrain.scf_fraction = scf_fraction
    cfg.mcmc.burn_in = 10
    cfg.optim.optimizer = optimizer
    cfg.optim.laplacian = laplacian
    cfg.optim.iterations = 3
    cfg.debug.check_nan = True
    cfg.observables.s2 = True
    cfg.observables.dipole = True
    cfg.observables.density = True
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)

  @parameterized.parameters([{'states': 0}, {'states': 3}])
  def test_s2_energy(self, states):
    cfg = diatomic.get_config()
    cfg.system.molecule_name = 'LiH'
    cfg.network.ferminet.hidden_dims = ((16, 4),) * 2
    cfg.network.determinants = 2
    cfg.batch_size = 32
    cfg.system.states = states
    cfg.pretrain.iterations = 10
    cfg.mcmc.burn_in = 10
    cfg.optim.iterations = 3
    cfg.optim.spin_energy = 1.0
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)

  def test_random_pretraining(self):
    cfg = diatomic.get_config()
    cfg.system.molecule_name = 'LiH'
    cfg.network.ferminet.hidden_dims = ((16, 4),) * 2
    cfg.network.determinants = 2
    cfg.batch_size = 32
    cfg.system.states = 2
    cfg.pretrain.iterations = 10
    cfg.pretrain.basis = 'ccpvdz'
    cfg.pretrain.excitation_type = 'random'
    cfg.mcmc.burn_in = 0
    cfg.optim.iterations = 0
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)

  def test_inference_step(self):
    cfg = diatomic.get_config()
    cfg.system.molecule_name = 'LiH'
    cfg.network.ferminet.hidden_dims = ((16, 4),) * 2
    cfg.network.determinants = 2
    cfg.batch_size = 32
    cfg.system.states = 2
    cfg.pretrain.iterations = 10
    cfg.mcmc.burn_in = 10
    cfg.optim.iterations = 3

    cfg.log.save_path = self.create_tempdir().full_path
    cfg.log.save_frequency = 0  # Save at every step.
    cfg = base_config.resolve(cfg)
    # Trivial training run
    train.train(cfg)

    # Update config and run inference
    cfg.optim.optimizer = 'none'
    cfg = base_config.resolve(cfg)
    train.train(cfg)

  def test_overlap_step(self):
    cfg = diatomic.get_config()
    cfg.system.molecule_name = 'LiH'
    cfg.network.ferminet.hidden_dims = ((16, 4),) * 2
    cfg.network.determinants = 2
    cfg.batch_size = 32
    cfg.system.states = 2
    cfg.pretrain.iterations = 10
    cfg.mcmc.burn_in = 10
    cfg.optim.iterations = 3
    cfg.optim.objective = 'vmc_overlap'
    cfg.debug.check_nan = True
    cfg.observables.s2 = True
    cfg.observables.dipole = True
    cfg.observables.density = True
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)


MOL_STRINGS = [
    'H 0 0 -1; H 0 0 1',
    'O 0 0 0; H  0 1 0; H 0 0 1',
    'N 0 0 0'  # Include a system with an odd number of electrons
]


class QmcPyscfMolTest(parameterized.TestCase):

  def setUp(self):
    super(QmcPyscfMolTest, self).setUp()
    # disable use of temp directory in pyscf.
    # Test calculations are small enough to fit in RAM and we don't need
    # checkpoint files.
    pyscf.lib.param.TMPDIR = None

  @parameterized.parameters(
      (mol_string, optimizer, mcmc_blocks)
      for mol_string, optimizer, mcmc_blocks in
      zip(MOL_STRINGS, ('adam', 'kfac'), (1, 2)))
  def test_training_step_pyscf(self, mol_string, optimizer, mcmc_blocks):
    mol = pyscf.gto.Mole()
    mol.build(
        atom=mol_string,
        basis='sto-3g', unit='bohr')
    cfg = base_config.default()
    cfg.system.pyscf_mol = mol

    cfg.network.ferminet.hidden_dims = ((16, 4),) * 2
    cfg.network.determinants = 2
    cfg.batch_size = 32
    cfg.pretrain.iterations = 10
    cfg.mcmc.burn_in = 10
    cfg.mcmc.blocks = mcmc_blocks
    cfg.optim.optimizer = optimizer
    cfg.optim.iterations = 3
    cfg.debug.check_nan = True
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)

  @parameterized.parameters([{'states': 0}, {'states': 3}])
  def test_pseudopotential_step(self, states):
    cfg = base_config.default()
    mol = pyscf.gto.Mole()
    mol.atom = ['Li 0 0 0']
    mol.basis = {'Li': cfg.system.pp.basis}
    mol.ecp = {'Li': cfg.system.pp.type}

    mol.charge = 0
    mol.spin = 1
    mol.unit = 'angstrom'
    mol.build()
    cfg.system.pyscf_mol = mol

    cfg.network.ferminet.hidden_dims = ((16, 4),) * 2
    cfg.network.determinants = 2
    cfg.batch_size = 32
    cfg.pretrain.iterations = 10
    cfg.mcmc.burn_in = 0
    cfg.system.use_pp = True
    cfg.system.pp.symbols = ['Li']
    cfg.system.states = states
    cfg.system.electrons = (1, 0)
    cfg.optim.iterations = 3
    cfg.debug.check_nan = True
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    # Calculation is too small to test the results for accuracy. Test just to
    # ensure they actually run without a top-level error.
    train.train(cfg)

  @parameterized.parameters(
      (False, (16, 16), (), 0, ()),
      (True, (16, 16), (), 0, ()),
      (False, (), (16, 16), 16, (16, 16)),
      (False, (16, 16), (16, 16), 16, (16, 16)),
      (True, (16, 16), (16, 16), 16, (16, 16)),
  )
  def test_schnet_training_step(
      self, split_spins, ee_dims, en_aux_dims, n_aux_dims, en_dims
  ):
    cfg = atom.get_config()
    cfg.system.atom = 'He'
    cfg.update_from_flattened_dict({
        'batch_size': 32,
        'pretrain.iterations': 10,
        'mcmc.burn_in': 10,
        'optim.iterations': 2,
        'network.determinants': 1,
        'network.ferminet.hidden_dims': ((16, 4), (16, 4)),
        'network.ferminet.separate_spin_channels': split_spins,
        'network.ferminet.schnet_electron_electron_convolutions': ee_dims,
        'network.ferminet.electron_nuclear_aux_dims': en_aux_dims,
        'network.ferminet.nuclear_embedding_dim': n_aux_dims,
        'network.ferminet.schnet_electron_nuclear_convolutions': en_dims,
        'log.save_path': self.create_tempdir().full_path,
    })
    cfg = base_config.resolve(cfg)
    train.train(cfg)


if __name__ == '__main__':
  absltest.main()
