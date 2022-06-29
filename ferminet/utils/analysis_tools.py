# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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

"""Tools for reading and analysing QMC data."""

from typing import Iterable, Optional, Union

from absl import logging
import numpy as np
import pandas as pd

from pyblock import pd_utils as blocking


def _format_network(network_option: Union[int, Iterable[int]]) -> str:
  """Formats a network configuration to a (short) string.

  Args:
    network_option: a integer or iterable of integers.

  Returns:
    String representation of the network option. If the network option is an
    iterable of the form [V, V, ...], return NxV, where N is the length of the
    iterable.
  """
  try:
    # pytype doesn't handle try...except TypeError gracefully.
    if all(xi == network_option[0] for xi in network_option[1:]):  # pytype: disable=unsupported-operands
      return f'{len(network_option)}x{network_option[0]}'  # pytype: disable=unsupported-operands,wrong-arg-types
    else:
      return str(network_option)
  except TypeError:
    return str(network_option)


def estimate_stats(df: pd.DataFrame,
                   burn_in: int,
                   groups: Optional[Iterable[str]] = None,
                   group_by_work_unit: bool = True) -> pd.DataFrame:
  """Estimates statistics for the (local) energy.

  Args:
    df: pd.DataFrame containing local energy data in the 'eigenvalues' column.
    burn_in: number of data points to discard before accumulating statistics to
      allow for learning and equilibration time.
    groups: list of column names in df to group by. The statistics for each
      group are returned, along with the corresponding data for the group.  The
      group columns should be sufficient to distinguish between separate work
      units/calculations in df.
    group_by_work_unit: add 'work_unit_id' to the list of groups if not already
      present and 'work_unit_id' is a column in df. This is usually helpful
      for safety, when each work unit is a separate calculation and should be
      treated separately statistically.

  Returns:
    pandas DataFrame containing estimates of the mean, standard error and error
    in the standard error from a blocking analysis of the local energy for each
    group in df.

  Raises:
    RuntimeError: If groups is empty or None and group_by_work_unit is False. If
    df does not contain a key to group over, insert a dummy column with
    identical values or use pyblock directly.
  """
  wid = 'work_unit_id'
  if groups is None:
    groups = []
  else:
    groups = list(groups)
  if group_by_work_unit and wid not in groups and wid in df.columns:
    groups.append(wid)
  if not groups:
    raise RuntimeError(
        'Must group by at least one key or set group_by_work_unit to True.')
  if len(groups) == 1:
    index_dict = {'index': groups[0]}
  else:
    index_dict = {f'level_{i}': group for i, group in enumerate(groups)}
  stats_dict = {
      'mean': 'energy',
      'standard error': 'stderr',
      'standard error error': 'stderrerr'
  }
  def block(key, values):
    blocked = blocking.reblock_summary(blocking.reblock(values)[1])
    if not blocked.empty:
      return blocked.iloc[0]
    else:
      logging.warning('Reblocking failed to estimate statistics for %s.', key)
      return pd.Series({statistic: np.nan for statistic in stats_dict})
  stats = (
      pd.DataFrame.from_dict({
          n: block(n, d.eigenvalues[burn_in:])
          for n, d in df.groupby(groups) if not d[burn_in:].eigenvalues.empty
      }, orient='index')
      .reset_index()
      .rename(index_dict, axis=1)
      .rename(stats_dict, axis=1)
  )
  stats = stats.sort_values(by=groups).reset_index(drop=True)
  stats['burn_in'] = burn_in
  return stats
