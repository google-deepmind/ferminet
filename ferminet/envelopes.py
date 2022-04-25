# Copyright 2022 DeepMind Technologies Limited.
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

"""Multiplicative envelope functions."""

import enum
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import attr
from ferminet import curvature_tags_and_blocks
from ferminet import network_blocks
from ferminet import sto
from ferminet.utils import scf
import jax
import jax.numpy as jnp
from typing_extensions import Protocol

_MAX_POLY_ORDER = 5  # highest polynomial used in envelopes


class EnvelopeType(enum.Enum):
  """The point at which the envelope is applied."""
  PRE_ORBITAL = enum.auto()
  PRE_DETERMINANT = enum.auto()
  POST_DETERMINANT = enum.auto()


class EnvelopeLabel(enum.Enum):
  """Available multiplicative envelope functions."""
  ISOTROPIC = enum.auto()
  DIAGONAL = enum.auto()
  FULL = enum.auto()
  NULL = enum.auto()
  STO = enum.auto()
  STO_POLY = enum.auto()
  OUTPUT = enum.auto()
  EXACT_CUSP = enum.auto()


class EnvelopeInit(Protocol):

  def __call__(
      self,
      natom: int,
      output_dims: Union[int, Sequence[int]],
      hf: Optional[scf.Scf],
      ndim: int) -> Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]:
    """Returns the envelope parameters.

    Envelopes applied separately to each spin channel must create a sequence of
    parameters, one for each spin channel. Other envelope types must create a
    single mapping.

    Args:
      natom: Number of atoms in the system.
      output_dims: The dimension of the layer to which the envelope is applied,
        per-spin channel for pre_determinant envelopes and a scalar otherwise.
      hf: If present, initialise the parameters to match the Hartree-Fock
        solution. Otherwise a random initialisation is use. Not supported by all
        envelope types.
      ndim: Dimension of system. Change with care.
    """


class EnvelopeApply(Protocol):

  def __call__(self, *, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
               **kwargs: jnp.ndarray) -> jnp.ndarray:
    """Returns a multiplicative envelope to ensure boundary conditions are met.

    If the envelope is applied before orbital shaping or after determinant
    evaluation, the envelope function is called once and N is the number of
    electrons. If the envelope is applied after orbital shaping and before
    determinant evaluation, the envelope function is called once per spin
    channel and N is the number of electrons in the spin channel.

    The envelope applied post-determinant evaluation is assumed to be in
    log-space.

    Args:
      ae: atom-electron vectors, shape (N, natom, ndim).
      r_ae: atom-electron distances, shape (N, natom, 1).
      r_ee: electron-electron distances, shape (N, nel, 1).
      **kwargs: learnable parameters of the envelope function.
    """


@attr.s(auto_attribs=True)
class Envelope:
  apply_type: EnvelopeType
  init: EnvelopeInit
  apply: EnvelopeApply


def _apply_covariance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Equivalent to jnp.einsum('ijk,kmjn->ijmn', x, y)."""
  # We can avoid first reshape - just make params['sigma'] rank 3
  i, _, _ = x.shape
  k, m, j, n = y.shape
  x = x.transpose((1, 0, 2))
  y = y.transpose((2, 0, 1, 3)).reshape((j, k, m * n))
  vdot = jax.vmap(jnp.dot, (0, 0))
  return vdot(x, y).reshape((j, i, m, n)).transpose((1, 0, 2, 3))


def make_isotropic_envelope() -> Envelope:
  """Creates an isotropic exponentially decaying multiplicative envelope."""

  def init(natom: int, output_dims: Sequence[int], hf: Optional[scf.Scf] = None,
           ndim: int = 3) -> Sequence[Mapping[str, jnp.ndarray]]:
    del hf, ndim  # unused
    params = []
    for output_dim in output_dims:
      params.append({
          'pi': jnp.ones(shape=(natom, output_dim)),
          'sigma': jnp.ones(shape=(natom, output_dim))
      })
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Computes an isotropic exponentially-decaying multiplicative envelope."""
    del ae, r_ee  # unused
    return jnp.sum(jnp.exp(-r_ae * sigma) * pi, axis=1)

  return Envelope(EnvelopeType.PRE_DETERMINANT, init, apply)


def make_diagonal_envelope() -> Envelope:
  """Creates a diagonal exponentially-decaying multiplicative envelope."""

  def init(natom: int, output_dims: Sequence[int], hf: Optional[scf.Scf] = None,
           ndim: int = 3) -> Sequence[Mapping[str, jnp.ndarray]]:
    del hf  # unused
    params = []
    for output_dim in output_dims:
      params.append({
          'pi': jnp.ones(shape=(natom, output_dim)),
          'sigma': jnp.ones(shape=(natom, ndim, output_dim))
      })
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Computes a diagonal exponentially-decaying multiplicative envelope."""
    del r_ae, r_ee  # unused
    r_ae_sigma = jnp.linalg.norm(ae[..., None] * sigma, axis=2)
    return jnp.sum(jnp.exp(-r_ae_sigma) * pi, axis=1)

  return Envelope(EnvelopeType.PRE_DETERMINANT, init, apply)


def make_full_envelope() -> Envelope:
  """Computes a fully anisotropic exponentially-decaying envelope."""

  def init(natom: int, output_dims: Sequence[int], hf: Optional[scf.Scf] = None,
           ndim: int = 3) -> Sequence[Mapping[str, jnp.ndarray]]:
    del hf  # unused
    eye = jnp.eye(ndim)
    params = []
    for output_dim in output_dims:
      params.append({
          'pi': jnp.ones(shape=(natom, output_dim)),
          'sigma': jnp.tile(eye[..., None, None], [1, 1, natom, output_dim])
      })
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Computes a fully anisotropic exponentially-decaying envelope."""
    del r_ae, r_ee  # unused
    ae_sigma = _apply_covariance(ae, sigma)
    ae_sigma = curvature_tags_and_blocks.register_qmc(
        ae_sigma, ae, sigma, type='full')
    r_ae_sigma = jnp.linalg.norm(ae_sigma, axis=2)
    return jnp.sum(jnp.exp(-r_ae_sigma) * pi, axis=1)

  return Envelope(EnvelopeType.PRE_DETERMINANT, init, apply)


def make_null_envelope() -> Envelope:
  """Creates an no-op (identity) envelope."""

  def init(natom: int, output_dims: Sequence[int], hf: Optional[scf.Scf] = None,
           ndim: int = 3) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim, hf  # unused
    return [{} for _ in output_dims]

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray,
            r_ee: jnp.ndarray) -> jnp.ndarray:
    del ae, r_ae, r_ee
    return jnp.ones(shape=(1,))

  return Envelope(EnvelopeType.PRE_DETERMINANT, init, apply)


def make_sto_envelope() -> Envelope:
  """Creates a Slater-type orbital envelope: exp(-sigma*r_ae) * r_ae^n * pi."""

  def init(natom: int, output_dims: int, hf: Optional[scf.Scf] = None,
           ndim: int = 3) -> Mapping[str, jnp.ndarray]:

    pi = jnp.zeros(shape=(natom, output_dims))
    sigma = jnp.tile(jnp.eye(ndim)[..., None, None], [1, 1, natom, output_dims])
    # log order of the polynomial (initialize so the order is near zero)
    n = -50 * jnp.ones(shape=(natom, output_dims))

    if hf is not None:
      j = 0
      for i, atom in enumerate(hf.molecule):
        coeffs = sto.STO_6G_COEFFS[atom.symbol]
        for orb in coeffs.keys():
          order = int(orb[0]) - (1 if orb[1] == 's' else 2)
          log_order = jnp.log(order + jnp.exp(-50.0))
          zeta, c = coeffs[orb]
          for _ in range(1 if orb[1] == 's' else 3):
            pi = pi.at[i, j].set(c)
            n = n.at[i, j].set(log_order)
            sigma = sigma.at[..., i, j].mul(zeta)
            j += 1
    return {'pi': pi, 'sigma': sigma, 'n': n}

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """Computes a Slater-type orbital envelope: exp(-sigma*r_ae) * r_ae^n * pi."""
    del r_ae, r_ee  # unused
    ae_sigma = _apply_covariance(ae, sigma)
    ae_sigma = curvature_tags_and_blocks.register_qmc(
        ae_sigma, ae, sigma, type='full')
    r_ae_sigma = jnp.linalg.norm(ae_sigma, axis=2)
    exp_r_ae = jnp.exp(-r_ae_sigma + jnp.exp(n) * jnp.log(r_ae_sigma))
    out = jnp.sum(exp_r_ae * pi, axis=1)
    return out

  return Envelope(EnvelopeType.PRE_ORBITAL, init, apply)


def make_sto_poly_envelope() -> Envelope:
  """Creates a Slater-type orbital envelope."""

  def init(natom: int, output_dims: int, hf: Optional[scf.Scf] = None,
           ndim: int = 3) -> Mapping[str, jnp.ndarray]:

    pi = jnp.zeros(shape=(natom, output_dims, _MAX_POLY_ORDER))
    sigma = jnp.tile(jnp.eye(ndim)[..., None, None], [1, 1, natom, output_dims])

    if hf is not None:
      # Initialize envelope to match basis set elements.
      j = 0
      for i, atom in enumerate(hf.molecule):
        coeffs = sto.STO_6G_COEFFS[atom.symbol]
        for orb in coeffs.keys():
          order = int(orb[0]) - (1 if orb[1] == 's' else 2)
          zeta, c = coeffs[orb]
          for _ in range(1 if orb[1] == 's' else 3):
            pi = pi.at[i, j, order].set(c)
            sigma = sigma.at[..., i, j].mul(zeta)
            j += 1
    return {'pi': pi, 'sigma': sigma}

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Computes a Slater-type orbital envelope."""
    del r_ae, r_ee  # unused
    # Should register KFAC tags and blocks.
    # Envelope: exp(-sigma*r_ae) * (sum_i r_ae^i * pi_i)
    ae_sigma = _apply_covariance(ae, sigma)
    ae_sigma = curvature_tags_and_blocks.register_qmc(
        ae_sigma, ae, sigma, type='full')
    r_ae_sigma = jnp.linalg.norm(ae_sigma, axis=2)
    exp_r_ae = jnp.exp(-r_ae_sigma)
    poly_r_ae = jnp.power(
        jnp.expand_dims(r_ae_sigma, -1), jnp.arange(_MAX_POLY_ORDER))
    out = jnp.sum(exp_r_ae * jnp.sum(poly_r_ae * pi, axis=3), axis=1)
    return out

  return Envelope(EnvelopeType.PRE_ORBITAL, init, apply)


def make_output_envelope() -> Envelope:
  """Creates an anisotropic multiplicative envelope to apply to determinants."""

  def init(natom: int, output_dims: int, hf: Optional[scf.Scf] = None,
           ndim: int = 3) -> Mapping[str, jnp.ndarray]:
    """Initialise learnable parameters for output envelope."""
    del output_dims, hf  # unused
    return {
        'pi': jnp.zeros(shape=natom),
        'sigma': jnp.tile(jnp.eye(ndim)[..., None], [1, 1, natom])
    }

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Fully anisotropic envelope, but only one output in log space."""
    del r_ae, r_ee  # unused
    # Should register KFAC tags and blocks.
    sigma = jnp.expand_dims(sigma, -1)
    ae_sigma = jnp.squeeze(_apply_covariance(ae, sigma), axis=-1)
    r_ae_sigma = jnp.linalg.norm(ae_sigma, axis=2)
    return jnp.sum(jnp.log(jnp.sum(jnp.exp(-r_ae_sigma + pi), axis=1)))

  return Envelope(EnvelopeType.POST_DETERMINANT, init, apply)


def make_exact_cusp_envelope(nspins: Tuple[int, int],
                             charges: jnp.ndarray) -> Envelope:
  """Creates an envelope satisfying cusp conditions to apply to determinants."""

  def init(natom: int, output_dims: int, hf: Optional[scf.Scf] = None,
           ndim: int = 3) -> Mapping[str, jnp.ndarray]:
    """Initialise learnable parameters for the exact cusp envelope."""
    del output_dims, hf  # unused
    return {
        'pi': jnp.zeros(shape=natom),
        'sigma': jnp.tile(jnp.eye(ndim)[..., None], [1, 1, natom])
    }

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Combine exact cusp conditions and envelope on the output into one."""
    # No cusp at zero
    del r_ae  # unused
    # Should register KFAC tags and blocks.
    sigma = jnp.expand_dims(sigma, -1)
    ae_sigma = jnp.squeeze(_apply_covariance(ae, sigma), axis=-1)
    soft_r_ae = jnp.sqrt(jnp.sum(1. + ae_sigma**2, axis=2))
    env = jnp.sum(jnp.log(jnp.sum(jnp.exp(-soft_r_ae + pi), axis=1)))

    # atomic cusp
    r_ae = jnp.linalg.norm(ae, axis=2)
    a_cusp = jnp.sum(charges / (1. + r_ae))

    # electronic cusp
    spin_partitions = network_blocks.array_partitions(nspins)
    r_ees = [
        jnp.split(r, spin_partitions, axis=1)
        for r in jnp.split(r_ee, spin_partitions, axis=0)
    ]
    # Sum over same-spin electrons twice but different-spin once, which
    # cancels out the different factor of 1/2 and 1/4 in the cusps.
    e_cusp = (
        jnp.sum(1. / (1. + r_ees[0][0])) + jnp.sum(1. / (1. + r_ees[1][1])) +
        jnp.sum(1. / (1. + r_ees[0][1])))
    return env + a_cusp - 0.5 * e_cusp

  return Envelope(EnvelopeType.POST_DETERMINANT, init, apply)


def get_envelope(
    envelope_label: EnvelopeLabel,
    **kwargs: Any,
) -> Envelope:
  """Gets the desired multiplicative envelope function.

  Args:
    envelope_label: envelope function required.
    **kwargs: keyword arguments forwarded to the envelope.

  Returns:
    (envelope_type, envelope), where envelope_type describes when the envelope
    should be applied in the network and envelope is the envelope function.
  """
  envelope_builders = {
      EnvelopeLabel.STO: make_sto_envelope,
      EnvelopeLabel.STO_POLY: make_sto_poly_envelope,
      EnvelopeLabel.ISOTROPIC: make_isotropic_envelope,
      EnvelopeLabel.DIAGONAL: make_diagonal_envelope,
      EnvelopeLabel.FULL: make_full_envelope,
      EnvelopeLabel.NULL: make_null_envelope,
      EnvelopeLabel.OUTPUT: make_output_envelope,
      EnvelopeLabel.EXACT_CUSP: make_exact_cusp_envelope,
  }
  return envelope_builders[envelope_label](**kwargs)
