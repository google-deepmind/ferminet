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
"""Writer utility classes."""

import contextlib
import os
from typing import Mapping, Optional, Sequence

from absl import logging
import tables


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
    self._file = open(self._filename, 'w')
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
        raise ValueError('Not a recognized key for writer: %s' % key)

    # write the data to csv
    self._file.write(','.join(row) + '\n')

    # write the data to abseil logs
    if self._log:
      logging.info('Iteration %s: %s', t, data)

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._file.close()


class H5Writer(contextlib.AbstractContextManager):
  """Write data to HDF5 files."""

  def __init__(self,
               name: str,
               schema: Mapping[str, Sequence[int]],
               directory: str = '',
               index_key: str = 't',
               compression_level: int = 5):
    """Initialise H5Writer.

    Args:
      name: file name for CSV.
      schema: dict of keys, corresponding to each data item . All data is
        assumed ot be 32-bit floats.
      directory: directory path to write file to.
      index_key: name of (integer) key used to index each entry.
      compression_level: compression level (0-9) used to compress HDF5 file.
    """
    self._path = os.path.join(directory, name)
    self._schema = schema
    self._index_key = index_key
    self._description = {}
    self._file = None
    self._complevel = compression_level

  def __enter__(self):
    if not self._schema:
      return self
    pos = 1
    self._description[self._index_key] = tables.Int32Col(pos=pos)
    for key, shape in self._schema.items():
      pos += 1
      self._description[key] = tables.Float32Col(pos=pos, shape=shape)
    if not os.path.isdir(os.path.dirname(self._path)):
      os.mkdir(os.path.dirname(self._path))
    self._file = tables.open_file(
        self._path,
        mode='w',
        title='Fermi Net Data',
        filters=tables.Filters(complevel=self._complevel))
    self._table = self._file.create_table(
        where=self._file.root, name='data', description=self._description)
    return self

  def write(self, index: int, data):
    """Write data to HDF5 file.

    Args:
      index: iteration index.
      data: dict of arrays to write to file. Only elements with keys in the
        schema are written.
    """
    if self._file:
      h5_data = (index,)
      for key in self._description:
        if key != self._index_key:
          h5_data += (data[key],)
      self._table.append([h5_data])
      self._table.flush()

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self._file:
      self._file.close()
      self._file = None
