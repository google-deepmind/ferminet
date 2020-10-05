# Lint as: python3
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

"""Extension to KFAC that adds correction term for non-normalized densities."""

import kfac


ip_p = kfac.utils.ip_p
sum_p = kfac.utils.sum_p
sprod_p = kfac.utils.sprod_p


class MeanCorrectedKfacOpt(kfac.PeriodicInvCovUpdateKfacOpt):
  """Like KFAC, but with all moments mean-centered."""

  def __init__(self, *args, **kwargs):
    """batch_size is mandatory, estimation_mode='exact' or 'exact_GGN'."""

    kwargs["compute_params_stats"] = True

    if not kwargs["estimation_mode"].startswith("exact"):  # pytype: disable=attribute-error
      raise ValueError("This is probably wrong.")

    super(MeanCorrectedKfacOpt, self).__init__(*args, **kwargs)

  def _multiply_preconditioner(self, vecs_and_vars):

    # Create a w list with the same order as vecs_and_vars
    w_map = {var: ps for (ps, var) in zip(self.params_stats,
                                          self.registered_variables)}
    w = tuple((w_map[var], var) for (_, var) in vecs_and_vars)

    r = super(MeanCorrectedKfacOpt, self)._multiply_preconditioner(
        vecs_and_vars)
    u = super(MeanCorrectedKfacOpt, self)._multiply_preconditioner(w)

    scalar = ip_p(w, r) / (1. + ip_p(w, w))

    return sum_p(r, sprod_p(scalar, u))
