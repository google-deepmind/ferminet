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
"""Writer utility classes."""

import contextlib
import os
from typing import Optional, Sequence

from absl import logging


class Writer(contextlib.AbstractContextManager):
  """Write data to CSV, as well as logging data to stdout if desired."""

  def __init__(self,
               name: str,
               schema: Sequence[str],
               directory: str = 'logs/',
               iteration_key: Optional[str] = 't',
               log: bool = True):
    """Initialise Writer.

    Args:
      name: file name for CSV.
      schema: sequence of keys, corresponding to each data item.
      directory: directory path to write file to.
      iteration_key: if not None or a null string, also include the iteration
        index as the first column in the CSV output with the given key.
      log: Also log each entry to stdout.
    """
    self._schema = schema
    if not os.path.isdir(directory):
      os.mkdir(directory)
    self._filename = os.path.join(directory, name + '.csv')
    self._iteration_key = iteration_key
    self._log = log

  def __enter__(self):
    self._file = open(self._filename, 'w', encoding='UTF-8')
    # write top row of csv
    if self._iteration_key:
      self._file.write(f'{self._iteration_key},')
    self._file.write(','.join(self._schema) + '\n')
    return self

  def write(self, t: int, **data):
    """Writes to file and stdout.

    Args:
      t: iteration index.
      **data: data items with keys as given in schema.
    """
    row = [str(data.get(key, '')) for key in self._schema]
    if self._iteration_key:
      row.insert(0, str(t))
    for key in data:
      if key not in self._schema:
        raise ValueError(f'Not a recognized key for writer: {key}')

    # write the data to csv
    self._file.write(','.join(row) + '\n')

    # write the data to abseil logs
    if self._log:
      logging.info('Iteration %s: %s', t, data)

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._file.close()
