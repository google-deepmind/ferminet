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

"""Tests for ferminet.utils.elements."""

from absl.testing import absltest
from absl.testing import parameterized

from ferminet.utils import elements


class ElementsTest(parameterized.TestCase):

  def test_elements(self):
    for n, element in elements.ATOMIC_NUMS.items():
      self.assertEqual(n, element.atomic_number)
      self.assertEqual(elements.SYMBOLS[element.symbol], element)
      if element.symbol == 'X':
        continue
      elif element.symbol in ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']:
        self.assertEqual(element.period, elements.ATOMIC_NUMS[n - 1].period + 1)
      elif element.symbol != 'H':
        self.assertEqual(element.period, elements.ATOMIC_NUMS[n - 1].period)
    self.assertCountEqual(
        (element.symbol for element in elements.ATOMIC_NUMS.values()),
        elements.SYMBOLS.keys())
    self.assertCountEqual(
        (element.atomic_number for element in elements.SYMBOLS.values()),
        elements.ATOMIC_NUMS.keys())

  @parameterized.parameters(
      (elements.SYMBOLS['H'], 1, 1, 1, 1, 0),
      (elements.SYMBOLS['He'], 1, 18, 0, 1, 1),
      (elements.SYMBOLS['Li'], 2, 1, 1, 2, 1),
      (elements.SYMBOLS['Be'], 2, 2, 0, 2, 2),
      (elements.SYMBOLS['C'], 2, 14, 2, 4, 2),
      (elements.SYMBOLS['N'], 2, 15, 3, 5, 2),
      (elements.SYMBOLS['Al'], 3, 13, 1, 7, 6),
      (elements.SYMBOLS['Zn'], 4, 12, 0, 15, 15),
      (elements.SYMBOLS['Ga'], 4, 13, 1, 16, 15),
      (elements.SYMBOLS['Kr'], 4, 18, 0, 18, 18),
      (elements.SYMBOLS['Ce'], 6, -1, -1, None, None),
      (elements.SYMBOLS['Ac'], 7, 3, -1, None, None),
  )
  def test_element_group_period(self, element, period, group, spin_config,
                                nalpha, nbeta):
    # Validate subset of elements. See below for more thorough tests using
    # properties of the periodic table.
    with self.subTest('Verify period'):
      self.assertEqual(element.period, period)
    with self.subTest('Verify group'):
      self.assertEqual(element.group, group)
    with self.subTest('Verify spin configuration'):
      if (element.period > 5 and
          (element.group == -1 or 3 <= element.group <= 12)):
        with self.assertRaises(NotImplementedError):
          _ = element.spin_config
      else:
        self.assertEqual(element.spin_config, spin_config)
    with self.subTest('Verify electrons per spin'):
      if (element.period > 5 and
          (element.group == -1 or 3 <= element.group <= 12)):
        with self.assertRaises(NotImplementedError):
          _ = element.nalpha
        with self.assertRaises(NotImplementedError):
          _ = element.nbeta
      else:
        self.assertEqual(element.nalpha, nalpha)
        self.assertEqual(element.nbeta, nbeta)

  def test_periods(self):
    self.assertLen(elements.ATOMIC_NUMS,
                   sum(len(period) for period in elements.PERIODS.values()))
    period_length = {0: 1, 1: 2, 2: 8, 3: 8, 4: 18, 5: 18, 6: 32, 7: 32}
    for p, es in elements.PERIODS.items():
      self.assertLen(es, period_length[p])

  def test_groups(self):
    # Atomic numbers of first element in each period.
    period_starts = sorted([
        period_elements[0].atomic_number
        for period_elements in elements.PERIODS.values()
    ])
    # Iterate over all elements in order of atomic number. Group should
    # increment monotonically (except for accommodating absence of d block and
    # presence of f block) and reset to 1 on the first element in each period.
    for i in range(1, len(elements.ATOMIC_NUMS)):
      element = elements.ATOMIC_NUMS[i]
      if element.atomic_number in period_starts:
        prev_group = 0
        fblock = 0
      if element.symbol == 'He':
        # Full shell, not just full s subshell.
        self.assertEqual(element.group, 18)
      elif element.group == -1:
        # Found a lanthanide (period 6) or actinide (period 7).
        self.assertIn(element.period, [6, 7])
        fblock += 1
      elif element.atomic_number == 5 or element.atomic_number == 13:
        # No d block (10 elements, groups 3-12) in periods 2 and 3.
        self.assertEqual(element.group, prev_group + 11)
      else:
        # Group should increment monotonically.
        self.assertEqual(element.group, prev_group + 1)
      if element.group != -1:
        prev_group = element.group
      self.assertGreaterEqual(prev_group, 1)
      self.assertLessEqual(prev_group, 18)
      if element.group == 4 and element.period > 6:
        # Should have seen 14 lanthanides (period 6) or 14 actinides (period 7).
        self.assertEqual(fblock, 14)

    # The periodic table (up to element 118) contains 7 periods.
    # Hydrogen and Helium are placed in groups 1 and 18 respectively.
    # Groups 1-2 (s-block) and 13-18 (p-block) are present in the second
    # period onwards, groups 3-12 (d-block) the fourth period onwards.
    # Check each group contains the expected number of elements.
    nelements_in_group = [0]*18
    for element in elements.ATOMIC_NUMS.values():
      if element.group != -1 and element.period != 0:
        nelements_in_group[element.group-1] += 1
    self.assertListEqual(nelements_in_group, [7, 6] + [4]*10 + [6]*5 + [7])


if __name__ == '__main__':
  absltest.main()
