# Copyright 2023 DeepMind Technologies Limited
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

"""Tools for downloading and processing pseudopotentials."""

from ferminet.utils import elements
import jax
import jax.numpy as jnp
import numpy as np
import pyscf


def gaussian(r, a, b, n):
  return a * r**n * jnp.exp(-b * r**2)


def eval_ecp(r, coeffs):
  r"""Evaluates r^2 U_l = \\sum A_{lk} r^{n_{lk}} e^{- B_{lk} r^2}."""
  val = 0.
  for r_exponent, v in enumerate(coeffs):
    for (exp_coeff, linear_coeff) in v:
      val += gaussian(r, linear_coeff, exp_coeff, r_exponent)
  return val


def calc_gaussian_cutoff(
    coeffs,
    v_tol,
    rmin=0.005,
    rmax=10,
    nr=10001,
):
  """Calculates the Gaussian cutoff."""
  r = np.linspace(rmin, rmax, nr)
  r_c_max = 0
  for r_exponent in range(len(coeffs)):
    for (exp_coeff, linear_coeff) in coeffs[r_exponent]:
      v = np.abs(gaussian(r, linear_coeff, exp_coeff, r_exponent)) / r**2
      try:
        ind = np.where(v > v_tol)[0][-1] + 1
      except IndexError:
        # in case all values are zero
        ind = 0
      r_c_new = r[ind]
      r_c_max = max(r_c_new, r_c_max)
  return r_c_max


def calc_r_c(ecp, v_tol, **kwargs):
  """Calculates r_c."""
  out = {}
  for l, coeffs in ecp:
    out[l] = calc_gaussian_cutoff(coeffs, v_tol, **kwargs)
  return out


def eval_ecp_on_grid(
    ecp_all,
    r_grid=None,
    log_r0=-5,
    log_rn=10,
    n_grid=10001,
):
  """r_grid overrules log_r0, log_rn and n_grid if set."""

  if r_grid is None:
    r_grid = jnp.logspace(log_r0, log_rn, n_grid)
  else:
    n_grid = r_grid.size

  n_channels = max(len(val[1]) for val in ecp_all.values())
  n_cores = {z: val[0] for z, val in ecp_all.items()}
  v_grid_dict = {}

  for z, (_, ecp_val) in ecp_all.items():
    v_grid = jnp.zeros((n_channels, n_grid))

    # zeff = z - n_cores[z]

    for l, coeffs in ecp_val:
      v_grid = v_grid.at[l].set(eval_ecp(r_grid, coeffs) / r_grid**2)

    v_grid_dict[z] = jnp.asarray(v_grid)

  return n_cores, v_grid_dict, r_grid, n_channels


def make_pp_args(symbols, v_tol=1e-5, quad_degree=4, zeta="d", return_ecp=True):
  """Creates arguments to use in pp_hamiltonian."""
  ecp_all = {}
  ecp_basis = {}
  for symb in symbols:
    z = elements.SYMBOLS[symb].atomic_number
    aug = ""
    ecp_symb = pyscf.gto.basis.load_ecp("ccecp", symb)
    ecp_basis_symb = pyscf.gto.basis.load(f"ccecp{aug}ccpv{zeta}z", symb)

    ecp_all[z] = ecp_symb
    ecp_basis[z] = ecp_basis_symb

  r_c = {
      z: max(calc_r_c(ecp_symb, v_tol).values())
      for z, (nelec, ecp_symb) in ecp_all.items()
  }

  n_cores, v_grid_dict, r_grid, n_channels = eval_ecp_on_grid(ecp_all)

  pp_args = n_cores, r_c, n_channels, v_grid_dict, r_grid, quad_degree
  if return_ecp:
    return (ecp_all, ecp_basis), pp_args
  else:
    return pp_args


def check_pp_args(pp_args):
  """Checks that pseudopotential arguments are consistent."""
  # TODO(hsutterud): switch to a better error type
  n_cores, r_c, n_channels, v_grid_dict, r_grid, quad_degree = pp_args

  assert n_cores.keys() == r_c.keys() == v_grid_dict.keys()
  assert isinstance(quad_degree, int)
  assert quad_degree > 0

  for _, v_grid in v_grid_dict.items():
    assert v_grid.shape[1] == r_grid.shape[0]
    assert v_grid.shape[0] <= n_channels


def leg_l0(x):
  return jnp.ones_like(x)


def leg_l1(x):
  return x


def leg_l2(x):
  return 0.5 * (3 * x**2 - 1)


def leg_l3(x):
  return 0.5 * (5 * x**3 - 3 * x)


def eval_leg(x, l):
  return jax.lax.switch(l, [leg_l0, leg_l1, leg_l2, leg_l3], x)
