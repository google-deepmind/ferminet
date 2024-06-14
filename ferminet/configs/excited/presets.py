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

"""Commonly used sets of parameters for running experiments.

These are common optimization settings or network architectures that are
designed to be applied on top of the base configuration from
base_config.get_base_config.

Usage:

  A flattened dict can be used directly to update a config, e.g.:

    config.update_from_flattened_dict(psiformer)
"""

from ferminet.utils import utils

psiformer = utils.flatten_dict_keys({
    'network': {
        'network_type': 'psiformer',
        'determinants': 16,
        'jastrow': 'simple_ee',
        'rescale_inputs': True,
        'psiformer': {
            'num_heads': 4,
            'mlp_hidden_dims': (256,),
            'num_layers': 4,
            'use_layer_norm': True,
        }
    }
})


ferminet = utils.flatten_dict_keys({
    'network': {
        'network_type': 'ferminet',
        'determinants': 16,
        'ferminet': {
            'hidden_dims': 4 * ((256, 32),)
        }
    }
})


excited_states = utils.flatten_dict_keys({
    'optim': {
        'clip_median': True,
        'reset_if_nan': True,
        'laplacian': 'folx',
    },
    'pretrain': {
        'basis': 'ccpvdz',
        'scf_fraction': 1.0
    }
})
