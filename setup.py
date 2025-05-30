# Copyright 2020 DeepMind Technologies Limited and Google LLC
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
# ============================================================================
"""Setup for pip package."""

import unittest

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'absl-py',
    'attrs',
    'chex',
    'h5py',
    'folx @ git+https://github.com/microsoft/folx',
    'jax',
    'jaxlib',
    # TODO(b/230487443) - use released version of kfac.
    'kfac_jax @ git+https://github.com/deepmind/kfac-jax',
    'ml-collections',
    'optax',
    'numpy',
    'pandas',
    'pyscf',
    'pyblock',
    'scipy',
    'typing_extensions',
]


def ferminet_test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover('ferminet/tests', pattern='*_test.py')
  return test_suite


setup(
    name='ferminet',
    version='0.2',
    description=(
        'A library to train networks to represent ground '
        'state wavefunctions of fermionic systems'
    ),
    url='https://github.com/deepmind/ferminet',
    author='DeepMind',
    author_email='no-reply@google.com',
    # Contained modules and scripts.
    entry_points={
        'console_scripts': [
            'ferminet = ferminet.main:main_wrapper',
        ],
    },
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={'testing': ['flake8', 'pylint', 'pytest', 'pytype']},
    platforms=['any'],
    license='Apache 2.0',
    test_suite='setup.ferminet_test_suite',
)
