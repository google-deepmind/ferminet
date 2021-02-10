# Copyright 2018 DeepMind Technologies Limited and Google LLC
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
    'absl-py', 'kfac>=0.2.3', 'numpy<1.19.0', 'pandas', 'pyscf', 'pyblock',
    'dm-sonnet<2.0', 'tables', 'tensorflow>=1.14,<2.0',
    'tensorflow_probability==0.8'
]
EXTRA_PACKAGES = {
    'tensorflow-gpu': ['tensorflow-gpu>=1.14,<2.0'],
}


def ferminet_test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover('ferminet/tests', pattern='*_test.py')
  return test_suite


setup(
    name='ferminet',
    version='0.1',
    description='A library to train networks to represent ground state wavefunctions of fermionic systems',
    url='https://github.com/deepmind/ferminet',
    author='DeepMind',
    author_email='no-reply@google.com',
    # Contained modules and scripts.
    scripts=['bin/ferminet'],
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
    test_suite='setup.ferminet_test_suite',
)
