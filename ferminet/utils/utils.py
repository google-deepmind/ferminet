# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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

"""Generic utils for all QMC calculations."""

from typing import Any, Callable, Mapping, Sequence


def select_output(f: Callable[..., Sequence[Any]],
                  argnum: int) -> Callable[..., Any]:
  """Return the argnum-th result from callable f."""

  def f_selected(*args, **kwargs):
    return f(*args, **kwargs)[argnum]

  return f_selected


def flatten_dict_keys(input_dict: Mapping[str, Any],
                      prefix: str = '') -> dict[str, Any]:
  """Flattens the keys of the given, potentially nested dictionary."""
  output_dict = {}
  for key, value in input_dict.items():
    nested_key = '{}.{}'.format(prefix, key) if prefix else key
    if isinstance(value, dict):
      output_dict.update(flatten_dict_keys(value, prefix=nested_key))
    else:
      output_dict[nested_key] = value
  return output_dict
