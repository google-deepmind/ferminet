# Lint as: python3
# Copyright 2018 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ferminet.train."""

import os

from absl import flags
from absl.testing import parameterized
from ferminet import train
from ferminet.utils import system
import numpy as np
import pyscf
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
# Default flags are sufficient so mark FLAGS as parsed so we can run the tests
# with py.test, which imports this file rather than runs it.
FLAGS.mark_as_parsed()


class MolVmcTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(MolVmcTest, self).setUp()
    # disable use of temp directory in pyscf.
    # Test calculations are small enough to fit in RAM and we don't need
    # checkpoint files.
    pyscf.lib.param.TMPDIR = None

  def _get_atoms(self, name, kwargs=None):
    configs = {
        'h2': lambda _: system.diatomic('H', 'H', bond_length=1.4),
        'helium': lambda _: system.atom('He'),
        'helium_triplet': lambda _: system.atom('He', spins=(2, 0)),
        # of each spin. Handle entirely spin-polarised systems.
        'hydrogen': lambda _: system.atom('H'),
        'lithium': lambda _: system.atom('Li'),
        'hn': lambda kwargs: system.hn(r=1.4, **kwargs),
    }
    return configs[name](kwargs)

  @parameterized.parameters(
      {
          'name': 'h2',
          'use_kfac': False
      },
      {
          'name': 'helium',
          'use_kfac': False
      },
      {
          'name': 'lithium',
          'use_kfac': False
      },
      {
          'name': 'helium',
          'use_kfac': False
      },
      {
          'name': 'helium_triplet',
          'use_kfac': False
      },
      {
          'name': 'hn',
          'args': {'n': 3},
          'use_kfac': False
      },
      {
          'name': 'hn',
          'args': {'n': 3},
          'use_kfac': True
      },
      {
          'name': 'hn',
          'args': {'n': 5},
          'use_kfac': True
      },
      {
          'name': 'hn',
          'args': {'n': 3},
          'use_kfac': True
      },
      {
          'name': 'hn',
          'args': {'n': 3},
          'use_kfac': False
      },
      {
          'name': 'hn',
          'args': {'n': 3},
          'use_kfac': True
      },
      {
          'name': 'hn',
          'args': {'n': 3},
          'use_kfac': True,
      },
  )
  def test_system(self,
                  name,
                  args=None,
                  use_kfac=True):
    atoms, electrons = self._get_atoms(name, args)
    train.train(
        atoms,
        electrons,
        batch_size=8,
        network_config=train.NetworkConfig(determinants=2),
        pretrain_config=train.PretrainConfig(iterations=20),
        optim_config=train.OptimConfig(iterations=2, use_kfac=use_kfac),
        mcmc_config=train.MCMCConfig(steps=2),
        logging_config=train.LoggingConfig(
            result_path=self.create_tempdir().full_path),
        multi_gpu=False)

  def test_restart(self):
    atoms, electrons = self._get_atoms('lithium')

    batch_size = 8
    network_config = train.NetworkConfig(
        determinants=2, hidden_units=((64, 16),) * 3)
    optim_config = train.OptimConfig(iterations=10, use_kfac=False)
    result_directory = self.create_tempdir().full_path

    # First run
    train.train(
        atoms,
        electrons,
        batch_size=batch_size,
        network_config=network_config,
        pretrain_config=train.PretrainConfig(iterations=20),
        optim_config=optim_config,
        mcmc_config=train.MCMCConfig(steps=10),
        logging_config=train.LoggingConfig(
            result_path=os.path.join(result_directory, 'run1')),
        multi_gpu=False)

    # Restart from first run
    # Already done pretraining and burn in, so just want to resume training
    # immediately.
    tf.reset_default_graph()
    train.train(
        atoms,
        electrons,
        batch_size=batch_size,
        network_config=network_config,
        pretrain_config=train.PretrainConfig(iterations=0),
        optim_config=optim_config,
        mcmc_config=train.MCMCConfig(burn_in=0, steps=10),
        logging_config=train.LoggingConfig(
            result_path=os.path.join(result_directory, 'run2'),
            restore_path=os.path.join(result_directory, 'run1', 'checkpoints')
        ),
        multi_gpu=False)


class AssignElectronsTest(parameterized.TestCase):

  def _expected_result(self, molecule, e_per_atom):
    nelectrons = [sum(spin_electrons) for spin_electrons in zip(*e_per_atom)]
    if nelectrons[0] == nelectrons[1]:
      for electrons in e_per_atom:
        if electrons[0] < electrons[1]:
          flip = True
          break
        elif electrons[0] > electrons[1]:
          flip = False
          break
      if flip:
        e_per_atom = [electrons[::-1] for electrons in e_per_atom]
    return np.concatenate(
        [np.tile(atom.coords, e[0]) for atom, e in zip(molecule, e_per_atom)] +
        [np.tile(atom.coords, e[1]) for atom, e in zip(molecule, e_per_atom)])

  @parameterized.parameters(
      {
          'molecule': [('O', 0, 0, 0)],
          'electrons': (5, 3),
          'electrons_per_atom': ((5, 3),),
      },
      {
          'molecule': [('O', 0, 0, 0), ('N', 1.2, 0, 0)],
          'electrons': (8, 7),
          'electrons_per_atom': ((3, 5), (5, 2)),
      },
      {
          'molecule': [('O', 0, 0, 0), ('N', 1.2, 0, 0)],
          'electrons': (7, 8),
          'electrons_per_atom': ((5, 3), (2, 5)),
      },
      {
          'molecule': [('O', 0, 0, 0), ('N', 1.2, 0, 0)],
          'electrons': (9, 7),
          'electrons_per_atom': ((4, 5), (5, 2)),
      },
      {
          'molecule': [('O', 0, 0, 0), ('N', 1.2, 0, 0)],
          'electrons': (10, 7),
          'electrons_per_atom': ((5, 5), (5, 2)),
      },
      {
          'molecule': [('O', 0, 0, 0), ('N', 1.2, 0, 0)],
          'electrons': (7, 7),
          'electrons_per_atom': ((5, 3), (2, 4)),
      },
      {
          'molecule': [('O', 0, 0, 0), ('O', 1.2, 0, 0)],
          'electrons': (9, 7),
          'electrons_per_atom': ((4, 4), (5, 3)),
      },
      {
          'molecule': [('O', 0, 0, 0), ('O', 20, 0, 0)],
          'electrons': (10, 6),
          'electrons_per_atom': ((5, 3), (5, 3)),
      },
      {
          'molecule': [('H', 0, 0, 0), ('H', 1.2, 0, 0)],
          'electrons': (1, 1),
          'electrons_per_atom': ((1, 0), (0, 1)),
      },
      {
          'molecule': [('B', 0, 0, 0), ('H', 1.2, 0, 0), ('H', -0.6, 0.6, 0),
                       ('H', -0.6, -0.6, 0)],
          'electrons': (4, 4),
          'electrons_per_atom': ((3, 2), (0, 1), (1, 0), (0, 1)),
      },
      {
          'molecule': [('B', 0, 0, 0), ('H', 1.2, 0, 0)],
          'electrons': (3, 2),
          'electrons_per_atom': ((3, 2), (0, 0)),
      },
      {
          'molecule': [('B', 0, 0, 0), ('H', 1.2, 0, 0), ('H', -0.6, 0.6, 0)],
          'electrons': (3, 2),
          'electrons_per_atom': ((3, 2), (0, 0), (0, 0)),
      },
  )
  def test_assign_electrons(self, molecule, electrons, electrons_per_atom):
    molecule = [system.Atom(v[0], v[1:]) for v in molecule]
    e_means = train.assign_electrons(molecule, electrons)
    expected_e_means = self._expected_result(molecule, electrons_per_atom)
    np.testing.assert_allclose(e_means, expected_e_means)

    e_means = train.assign_electrons(molecule[::-1], electrons)
    if any(atom.symbol != molecule[0].symbol for atom in molecule):
      expected_e_means = self._expected_result(molecule[::-1],
                                               electrons_per_atom[::-1])
    else:
      expected_e_means = self._expected_result(molecule[::-1],
                                               electrons_per_atom)
    np.testing.assert_allclose(e_means, expected_e_means)


if __name__ == '__main__':
  tf.test.main()
