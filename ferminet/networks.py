# Copyright 2020 DeepMind Technologies Limited.
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

"""Implementation of Fermionic Neural Network in JAX."""
import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

from ferminet import curvature_tags_and_blocks
from ferminet import sto
from ferminet.utils import scf
import jax
import jax.numpy as jnp

_MAX_POLY_ORDER = 5  # highest polynomial used in envelopes


FermiLayers = Tuple[Tuple[int, int], ...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]
# pytype: enable=not-supported-yet
# init(key) -> params
FermiNetInit = Callable[[jnp.ndarray], ParamTree]
# network(params, x) -> sign_out, log_out
FermiNetApply = Callable[[ParamTree, jnp.ndarray], Tuple[jnp.ndarray,
                                                         jnp.ndarray]]


def init_fermi_net_params(
    key: jnp.ndarray,
    atoms: jnp.ndarray,
    spins: Tuple[int, int],
    envelope_type: str = 'full',
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    hf_solution: Optional[scf.Scf] = None,
    eps: float = 0.01,
    full_det: bool = True,
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    after_determinants: Union[int, Tuple[int, ...]] = 1,
):
  """Initializes parameters for the Fermionic Neural Network.

  Args:
    key: JAX RNG state.
    atoms: (natom, 3) array of atom positions.
    spins: Tuple of the number of spin-up and spin-down electrons.
    envelope_type: Envelope to use to impose orbitals go to zero at infinity.
      See fermi_net_orbitals.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    use_last_layer: If true, the outputs of the one- and two-electron streams
      are combined into permutation-equivariant features and passed into the
      final orbital-shaping layer. Otherwise, just the output of the
      one-electron stream is passed into the orbital-shaping layer.
    hf_solution: If present, initialise the parameters to match the Hartree-Fock
      solution. Otherwise a random initialisation is use.
    eps: If hf_solution is present, scale all weights and biases except the
      first layer by this factor such that they are initialised close to zero.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    determinants: Number of determinants to use.
    after_determinants: currently ignored.

  Returns:
    PyTree of network parameters.
  """
  # after_det is from the legacy QMC TF implementation. Reserving for future
  # use.
  del after_determinants

  if envelope_type in ('sto', 'sto-poly'):
    if bias_orbitals: raise ValueError('Cannot bias orbitals w/STO envelope.')
  if hf_solution is not None:
    if use_last_layer: raise ValueError('Cannot use last layer w/HF init')
    if envelope_type not in ('sto', 'sto-poly'):
      raise ValueError('When using HF init, '
                       'envelope_type must be `sto` or `sto-poly`.')

  natom = atoms.shape[0]
  in_dims = (natom*4, 4)
  nchannels = sum(spin > 0 for spin in spins)
  # The input to layer L of the one-electron stream is from
  # construct_symmetric_features and shape (nelectrons, nfeatures), where
  # nfeatures is i) output from the previous one-electron layer; ii) the mean
  # for each spin channel from each layer; iii) the mean for each spin channel
  # from each two-electron layer. We don't create features for spin channels
  # which contain no electrons (i.e. spin-polarised systems).
  dims_one_in = (
      [(nchannels + 1) * in_dims[0] + nchannels * in_dims[1]] +
      [(nchannels + 1) * hdim[0] + nchannels * hdim[1] for hdim in hidden_dims])
  if not use_last_layer:
    dims_one_in[-1] = hidden_dims[-1][0]
  dims_one_out = [hdim[0] for hdim in hidden_dims]
  dims_two = [in_dims[1]] + [hdim[1] for hdim in hidden_dims]

  len_double = len(hidden_dims) if use_last_layer else len(hidden_dims) - 1
  params = {
      'single': [{} for i in range(len(hidden_dims))],
      'double': [{} for i in range(len_double)],
      'orbital': [],
      'envelope': [{} for spin in spins if spin > 0]
  }

  if envelope_type in ['output', 'exact_cusp']:
    params['envelope'] = {
        'pi': jnp.zeros(natom),
        'sigma': jnp.tile(jnp.eye(3)[..., None], [1, 1, natom])
    }
  elif envelope_type == 'sto':
    params['envelope'] = {
        'pi': jnp.zeros((natom, dims_one_in[-1])),
        'sigma': jnp.tile(jnp.eye(3)[..., None, None],
                          [1, 1, natom, dims_one_in[-1]]),
        # log order of the polynomial (initialize so the order is near zero)
        'n': -50 * jnp.ones((natom, dims_one_in[-1])),
    }
    if hf_solution is not None:
      j = 0
      for i, atom in enumerate(hf_solution.molecule):
        coeffs = sto.STO_6G_COEFFS[atom.symbol]
        for orb in coeffs.keys():
          order = int(orb[0]) - (1 if orb[1] == 's' else 2)
          log_order = jnp.log(order + jnp.exp(-50.0))
          zeta, c = coeffs[orb]
          for _ in range(1 if orb[1] == 's' else 3):
            pi = params['envelope']['pi'].at[i, j].set(c)
            n = params['envelope']['n'].at[i, j].set(log_order)
            sigma = params['envelope']['sigma'].at[..., i, j].mul(zeta)
            params['envelope'] = {'pi': pi, 'sigma': sigma, 'n': n}
            j += 1
  elif envelope_type == 'sto-poly':
    params['envelope'] = {
        'pi': jnp.zeros((natom, dims_one_in[-1], _MAX_POLY_ORDER)),
        'sigma': jnp.tile(jnp.eye(3)[..., None, None],
                          [1, 1, natom, dims_one_in[-1]]),
    }
    if hf_solution is not None:
      # Initialize envelope to match basis set elements.
      j = 0
      for i, atom in enumerate(hf_solution.molecule):
        coeffs = sto.STO_6G_COEFFS[atom.symbol]
        for orb in coeffs.keys():
          order = int(orb[0]) - (1 if orb[1] == 's' else 2)
          zeta, c = coeffs[orb]
          for _ in range(1 if orb[1] == 's' else 3):
            pi = params['envelope']['pi'].at[i, j, order].set(c)
            sigma = params['envelope']['sigma'].at[..., i, j].mul(zeta)
            params['envelope'] = {'pi': pi, 'sigma': sigma}
            j += 1
  else:
    params['envelope'] = [{} for spin in spins if spin > 0]
    for i, spin in enumerate((spin for spin in spins if spin > 0)):
      nparam = sum(spins)*determinants if full_det else spin*determinants
      params['envelope'][i]['pi'] = jnp.ones((natom, nparam))
      if envelope_type == 'isotropic':
        params['envelope'][i]['sigma'] = jnp.ones((natom, nparam))
      elif envelope_type == 'diagonal':
        params['envelope'][i]['sigma'] = jnp.ones((natom, 3, nparam))
      elif envelope_type == 'full':
        params['envelope'][i]['sigma'] = jnp.tile(
            jnp.eye(3)[..., None, None], [1, 1, natom, nparam])

  for i in range(len(hidden_dims)):
    key, subkey = jax.random.split(key)
    params['single'][i]['w'] = (jax.random.normal(
        subkey, shape=(dims_one_in[i], dims_one_out[i])) /
                                jnp.sqrt(float(dims_one_in[i])))

    key, subkey = jax.random.split(key)
    params['single'][i]['b'] = jax.random.normal(
        subkey, shape=(dims_one_out[i],))

    if hf_solution is not None:
      # Scale all params to be near zero except first layer (set below).
      params['single'][i]['w'] = params['single'][i]['w'] * eps
      params['single'][i]['b'] = params['single'][i]['b'] * eps

    if i < len_double:
      key, subkey = jax.random.split(key)
      params['double'][i]['w'] = (jax.random.normal(
          subkey, shape=(dims_two[i], dims_two[i+1])) /
                                  jnp.sqrt(float(dims_two[i])))

      key, subkey = jax.random.split(key)
      params['double'][i]['b'] = jax.random.normal(subkey,
                                                   shape=(dims_two[i+1],))

  if hf_solution is not None:
    # Initialize first layer of Fermi Net to match s- or p-type orbitals.
    # The sto and sto-poly envelopes can exactly capture the s-type orbital,
    # so the effect of the neural network part is constant, while the p-type
    # orbital also has a term multiplied by x, y or z.
    j = 0
    for ia, atom in enumerate(hf_solution.molecule):
      coeffs = sto.STO_6G_COEFFS[atom.symbol]
      for orb in coeffs.keys():
        if orb[1] == 's':
          params['single'][0]['b'] = params['single'][0]['b'].at[j].set(1.0)
          j += 1
        elif orb[1] == 'p':
          w = params['single'][0]['w']
          w = w.at[ia*4+1:(ia+1)*4, j:j+3].set(jnp.eye(3))
          params['single'][0]['w'] = w
          j += 3
        else:
          raise NotImplementedError('HF Initialization not implemented for '
                                    '%s orbitals' % orb[1])

  for i, spin in enumerate((spin for spin in spins if spin > 0)):
    nparam = sum(spins)*determinants if full_det else spin*determinants
    key, subkey = jax.random.split(key)
    params['orbital'].append({})
    params['orbital'][i]['w'] = (jax.random.normal(
        subkey, shape=(dims_one_in[-1], nparam)) /
                                 jnp.sqrt(float(dims_one_in[-1])))
    if bias_orbitals:
      key, subkey = jax.random.split(key)
      params['orbital'][i]['b'] = jax.random.normal(
          subkey, shape=(nparam,))
    if hf_solution is not None:
      # Initialize last layer to match Hartree-Fock weights on basis set.
      # pylint: disable=protected-access
      norb = hf_solution._mean_field.mo_coeff[i].shape[0]
      params['orbital'][i]['w'] = params['orbital'][i]['w'] * eps
      mat = hf_solution._mean_field.mo_coeff[i][:, :spin]
      # pylint: enable=protected-access
      w = params['orbital'][i]['w']
      for j in range(determinants):
        w = w.at[:norb, j*spin:(j+1)*spin].set(mat)
      params['orbital'][i]['w'] = w

  return params


def construct_input_features(
    x: jnp.ndarray,
    atoms: jnp.ndarray,
    ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Constructs inputs to Fermi Net from raw electron and atomic positions.

  Args:
    x: electron positions. Shape (nelectrons*ndim,).
    atoms: atom positions. Shape (natoms, ndim).
    ndim: dimension of system. Change only with caution.

  Returns:
    ae, ee, r_ae, r_ee tuple, where:
      ae: atom-electron vector. Shape (nelectron, natom, 3).
      ee: atom-electron vector. Shape (nelectron, nelectron, 3).
      r_ae: atom-electron distance. Shape (nelectron, natom, 1).
      r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
    The diagonal terms in r_ee are masked out such that the gradients of these
    terms are also zero.
  """
  assert atoms.shape[1] == ndim
  ae = jnp.reshape(x, [-1, 1, ndim]) - atoms[None, ...]
  ee = jnp.reshape(x, [1, -1, ndim]) - jnp.reshape(x, [-1, 1, ndim])

  r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
  # Avoid computing the norm of zero, as is has undefined grad
  n = ee.shape[0]
  r_ee = (
      jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))

  return ae, ee, r_ae, r_ee[..., None]


def construct_symmetric_features(h_one: jnp.ndarray, h_two: jnp.ndarray,
                                 spins: Tuple[int, int]) -> jnp.ndarray:
  """Combines intermediate features from rank-one and -two streams.

  Args:
    h_one: set of one-electron features. Shape: (nelectrons, n1), where n1 is
      the output size of the previous layer.
    h_two: set of two-electron features. Shape: (nelectrons, nelectrons, n2),
      where n2 is the output size of the previous layer.
    spins: number of spin-up and spin-down electrons.

  Returns:
    array containing the permutation-equivariant features: the input set of
    one-electron features, the mean of the one-electron features over each
    (occupied) spin channel, and the mean of the two-electron features over each
    (occupied) spin channel. Output shape (nelectrons, 3*n1 + 2*n2) if there are
    both spin-up and spin-down electrons and (nelectrons, 2*n1, n2) otherwise.
  """
  # Split features into spin up and spin down electrons
  h_ones = jnp.split(h_one, spins[0:1], axis=0)
  h_twos = jnp.split(h_two, spins[0:1], axis=0)

  # Construct inputs to next layer
  # h.size == 0 corresponds to unoccupied spin channels.
  g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
  g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]

  g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]

  return jnp.concatenate([h_one] + g_one + g_two, axis=1)


def isotropic_envelope(ae, params):
  """Computes an isotropic exponentially-decaying multiplicative envelope."""
  return jnp.sum(jnp.exp(-ae * params['sigma']) * params['pi'], axis=1)


def diagonal_envelope(ae, params):
  """Computes a diagonal exponentially-decaying multiplicative envelope."""
  r_ae = jnp.linalg.norm(ae[..., None] * params['sigma'], axis=2)
  return jnp.sum(jnp.exp(-r_ae) * params['pi'], axis=1)


vdot = jax.vmap(jnp.dot, (0, 0))


def apply_covariance(x, y):
  """Equivalent to jnp.einsum('ijk,kmjn->ijmn', x, y)."""
  i, _, _ = x.shape
  k, m, j, n = y.shape
  x = x.transpose((1, 0, 2))
  y = y.transpose((2, 0, 1, 3)).reshape((j, k, m*n))
  return vdot(x, y).reshape((j, i, m, n)).transpose((1, 0, 2, 3))


def full_envelope(ae, params):
  """Computes a fully anisotropic exponentially-decaying multiplicative envelope."""
  r_ae = apply_covariance(ae, params['sigma'])
  r_ae = curvature_tags_and_blocks.register_qmc1(r_ae, ae, params['sigma'],
                                                 type='full')
  r_ae = jnp.linalg.norm(r_ae, axis=2)
  return jnp.sum(jnp.exp(-r_ae) * params['pi'], axis=1)


def sto_envelope(ae, params):
  """Computes a Slater-type orbital envelope: exp(-sigma*r_ae) * r_ae^n * pi."""
  r_ae = apply_covariance(ae, params['sigma'])
  r_ae = curvature_tags_and_blocks.register_qmc1(r_ae, ae, params['sigma'],
                                                 type='full')
  r_ae = jnp.linalg.norm(r_ae, axis=2)
  exp_r_ae = jnp.exp(-r_ae + jnp.exp(params['n']) * jnp.log(r_ae))
  out = jnp.sum(exp_r_ae * params['pi'], axis=1)
  # return curvature_tags_and_blocks.register_qmc2(out, exp_r_ae, params['pi'])
  return out


def sto_poly_envelope(ae, params):
  """Computes a Slater-type orbital envelope: exp(-sigma*r_ae) * (sum_i r_ae^i * pi_i)."""
  r_ae = apply_covariance(ae, params['sigma'])
  r_ae = curvature_tags_and_blocks.register_qmc1(r_ae, ae, params['sigma'],
                                                 type='full')
  r_ae = jnp.linalg.norm(r_ae, axis=2)
  exp_r_ae = jnp.exp(-r_ae)
  poly_r_ae = jnp.power(jnp.expand_dims(r_ae, -1), jnp.arange(_MAX_POLY_ORDER))
  out = jnp.sum(exp_r_ae * jnp.sum(poly_r_ae * params['pi'], axis=3), axis=1)
  # return curvature_tags_and_blocks.register_qmc2(out, exp_r_ae, params['pi'])
  return out


def output_envelope(ae, params):
  """Fully anisotropic envelope, but only one output."""
  sigma = jnp.expand_dims(params['sigma'], -1)
  ae_sigma = jnp.squeeze(apply_covariance(ae, sigma), axis=-1)
  r_ae = jnp.linalg.norm(ae_sigma, axis=2)
  return jnp.sum(jnp.log(jnp.sum(jnp.exp(-r_ae + params['pi']), axis=1)))


def exact_cusp_envelope(ae, r_ee, params, charges, spins):
  """Combine exact cusp conditions and envelope on the output into one."""
  # No cusp at zero
  sigma = jnp.expand_dims(params['sigma'], -1)
  ae_sigma = jnp.squeeze(apply_covariance(ae, sigma), axis=-1)
  soft_r_ae = jnp.sqrt(jnp.sum(1. + ae_sigma**2, axis=2))
  env = jnp.sum(jnp.log(jnp.sum(jnp.exp(-soft_r_ae + params['pi']), axis=1)))

  # atomic cusp
  r_ae = jnp.linalg.norm(ae, axis=2)
  a_cusp = jnp.sum(charges / (1. + r_ae))

  # electronic cusp
  r_ees = [jnp.split(r, spins[0:1], axis=1)
           for r in jnp.split(r_ee, spins[0:1], axis=0)]
  # Sum over same-spin electrons twice but different-spin once, which
  # cancels out the different factor of 1/2 and 1/4 in the cusps.
  e_cusp = (jnp.sum(1. / (1. + r_ees[0][0])) +
            jnp.sum(1. / (1. + r_ees[1][1])) +
            jnp.sum(1. / (1. + r_ees[0][1])))
  return env + a_cusp - 0.5 * e_cusp


def slogdet(x):
  """Computes sign and log of determinants of matrices.

  This is a jnp.linalg.slogdet with a special (fast) path for small matrices.

  Args:
    x: square matrix.

  Returns:
    sign, (natural) logarithm of the determinant of x.
  """
  if x.shape[-1] == 1:
    sign = jnp.sign(x[..., 0, 0])
    logdet = jnp.log(jnp.abs(x[..., 0, 0]))
  else:
    sign, logdet = jnp.linalg.slogdet(x)

  return sign, logdet


def logdet_matmul(xs: Sequence[jnp.ndarray],
                  w: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Combines determinants and takes dot product with weights in log-domain.

  We use the log-sum-exp trick to reduce numerical instabilities.

  Args:
    xs: FermiNet orbitals in each determinant. Either of length 1 with shape
      (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
      (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
      determinants are factorised into block-diagonals for each spin channel).
    w: weight of each determinant. If none, a uniform weight is assumed.

  Returns:
    sum_i w_i D_i in the log domain, where w_i is the weight of D_i, the i-th
    determinant (or product of the i-th determinant in each spin channel, if
    full_det is not used).
  """
  slogdets = [slogdet(x) for x in xs]
  sign_in, logdet = functools.reduce(
      lambda a, b: (a[0]*b[0], a[1]+b[1]), slogdets)

  # log-sum-exp trick
  maxlogdet = jnp.max(logdet)
  det = sign_in * jnp.exp(logdet - maxlogdet)
  if w is None:
    result = jnp.sum(det)
  else:
    result = jnp.matmul(det, w)[0]
  sign_out = jnp.sign(result)
  log_out = jnp.log(jnp.abs(result)) + maxlogdet
  return sign_out, log_out


def linear_layer(x, w, b=None):
  """Evaluates a linear layer, x w + b.

  Args:
    x: inputs.
    w: weights.
    b: optional bias. Only x w is computed if b is None.

  Returns:
    x w + b if b is given, x w otherwise.
  """
  y = jnp.dot(x, w)
  y = y + b if b is not None else y
  return curvature_tags_and_blocks.register_repeated_dense(y, x, w, b)

vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)


def fermi_net_orbitals(params, x,
                       atoms=None,
                       spins=(None, None),
                       envelope_type=None,
                       full_det=True):
  """Forward evaluation of the Fermionic Neural Network up to the orbitals.

  Args:
    params: A dictionary of parameters, containing fields:
      `atoms`: atomic positions, used to construct input features.
      `single`: a list of dictionaries with params 'w' and 'b', weights for the
        one-electron stream of the network.
      `double`: a list of dictionaries with params 'w' and 'b', weights for the
        two-electron stream of the network.
      `orbital`: a list of two weight matrices, for spin up and spin down (no
        bias is necessary as it only adds a constant to each row, which does
        not change the determinant).
      `dets`: weight on the linear combination of determinants
      `envelope`: a dictionary with fields `sigma` and `pi`, weights for the
        multiplicative envelope.
    x: The input data, a 3N dimensional vector.
    atoms: Array with positions of atoms.
    spins: Tuple with number of spin up and spin down electrons.
    envelope_type: a string that specifies kind of envelope. One of:
      `isotropic`: envelope is the same in every direction
      `diagonal`: envelope has diagonal covariance
      `full`: envelope has full 3x3 covariance. Surprisingly memory inefficient!
      `sto`: A Slater-type orbital. Has additional polynomial term on top of
        exponential, and is applied *before* the orbitals. Used for initializing
        to the Hartree-Fock solution.
      `sto-poly`: Similar to `sto`, but with a more stable polynomial term.
      `output` or `exact_cusp`: ignored by this function
    full_det: If true, the determinants are dense, rather than block-sparse.
      True by default, false is still available for backward compatibility.
      Thus, the output shape of the orbitals will be (ndet, nalpha+nbeta,
      nalpha+nbeta) if True, and (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)
      if False.

  Returns:
    One (two matrices if full_det is False) that exchange columns under the
    exchange of inputs, and additional variables that may be needed by the
    envelope, depending on the envelope type.
  """

  ae_, ee_, r_ae, r_ee = construct_input_features(x, atoms)
  ae = jnp.concatenate((r_ae, ae_), axis=2)
  ae = jnp.reshape(ae, [jnp.shape(ae)[0], -1])
  ee = jnp.concatenate((r_ee, ee_), axis=2)

  # which variable do we pass to envelope?
  to_env = r_ae if envelope_type == 'isotropic' else ae_
  if envelope_type == 'exact_cusp':
    to_env = (to_env, r_ee)
  elif envelope_type == 'isotropic':
    envelope = isotropic_envelope
  elif envelope_type == 'diagonal':
    envelope = diagonal_envelope
  elif envelope_type == 'full':
    envelope = full_envelope
  elif envelope_type == 'sto':
    envelope = sto_envelope
  elif envelope_type == 'sto-poly':
    envelope = sto_poly_envelope

  h_one = ae  # single-electron features
  h_two = ee  # two-electron features
  residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
  for i in range(len(params['double'])):
    h_one_in = construct_symmetric_features(h_one, h_two, spins)

    # Execute next layer
    h_one_next = jnp.tanh(linear_layer(h_one_in, **params['single'][i]))
    h_two_next = jnp.tanh(vmap_linear_layer(h_two, params['double'][i]['w'],
                                            params['double'][i]['b']))
    h_one = residual(h_one, h_one_next)
    h_two = residual(h_two, h_two_next)
  if len(params['double']) != len(params['single']):
    h_one_in = construct_symmetric_features(h_one, h_two, spins)
    h_one_next = jnp.tanh(linear_layer(h_one_in, **params['single'][-1]))
    h_one = residual(h_one, h_one_next)
    h_to_orbitals = h_one
  else:
    h_to_orbitals = construct_symmetric_features(h_one, h_two, spins)
  if envelope_type in ('sto', 'sto-poly'):
    h_to_orbitals = envelope(to_env, params['envelope']) * h_to_orbitals
  # Note split creates arrays of size 0 for spin channels without any electrons.
  h_to_orbitals = jnp.split(h_to_orbitals, spins[0:1], axis=0)

  orbitals = [linear_layer(h, **p)
              for h, p in zip(h_to_orbitals, params['orbital'])]
  if envelope_type in ['isotropic', 'diagonal', 'full']:
    orbitals = [envelope(te, param)*orbital for te, orbital, param in
                zip(jnp.split(to_env, spins[0:1], axis=0),
                    orbitals, params['envelope'])]
  # Reshape into matrices and drop unoccupied spin channels.
  orbitals = [jnp.reshape(orbital, [spin, -1, sum(spins) if full_det else spin])
              for spin, orbital in zip(spins, orbitals) if spin > 0]
  orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
  if full_det:
    orbitals = [jnp.concatenate(orbitals, axis=1)]
  return orbitals, to_env


def fermi_net(params, x,
              atoms=None,
              spins=(None, None),
              charges=None,
              envelope_type='full',
              full_det=True):
  """Forward evaluation of the Fermionic Neural Network for a single datum.

  Args:
    params: A dictionary of parameters, containing fields:
      `atoms`: atomic positions, used to construct input features.
      `single`: a list of dictionaries with params 'w' and 'b', weights for the
        one-electron stream of the network.
      `double`: a list of dictionaries with params 'w' and 'b', weights for the
        two-electron stream of the network.
      `orbital`: a list of two weight matrices, for spin up and spin down (no
        bias is necessary as it only adds a constant to each row, which does
        not change the determinant).
      `dets`: weight on the linear combination of determinants
      `envelope`: a dictionary with fields `sigma` and `pi`, weights for the
        multiplicative envelope.
    x: The input data, a 3N dimensional vector.
    atoms: Array with positions of atoms.
    spins: Tuple with number of spin up and spin down electrons.
    charges: The charges of the atoms. Only needed if using exact cusps.
    envelope_type: a string that specifies kind of envelope. One of:
      `isotropic`: envelope is the same in every direction
      `diagonal`: envelope has diagonal covariance
      `full`: envelope has full 3x3 covariance. Surprisingly memory inefficient!
      `output`: same as `full`, but on the output of the network, not orbitals
      `exact_cusp`: same as `output`, but exactly satisfies cusp condition
    full_det: If true, the determinants are dense, rather than block-sparse.
      True by default, false is still available for backward compatibility.

  Returns:
    Output of antisymmetric neural network in log space, i.e. a tuple of sign of
    and log absolute of the network evaluated at x.
  """

  orbitals, to_env = fermi_net_orbitals(params, x,
                                        atoms=atoms,
                                        spins=spins,
                                        envelope_type=envelope_type,
                                        full_det=full_det)
  output = logdet_matmul(orbitals)
  if envelope_type == 'output':
    output = output[0], output[1] + output_envelope(to_env, params['envelope'])
  if envelope_type == 'exact_cusp':
    to_env, r_ee = to_env
    output = output[0], output[1] + exact_cusp_envelope(
        to_env, r_ee, params['envelope'], charges, spins)
  return output


def make_fermi_net(
    atoms: jnp.ndarray,
    spins: Tuple[int, int],
    charges: jnp.ndarray,
    envelope_type: str = 'full',
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    hf_solution: Optional[scf.Scf] = None,
    full_det: bool = True,
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    after_determinants: Union[int, Tuple[int, ...]] = 1,
) -> Tuple[FermiNetInit, FermiNetApply]:
  """Creates functions for initializing parameters and evaluating ferminet.

  Args:
    atoms: (natom, 3) array of atom positions.
    spins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    envelope_type: Envelope to use to impose orbitals go to zero at infinity.
      See fermi_net_orbitals.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    use_last_layer: If true, the outputs of the one- and two-electron streams
      are combined into permutation-equivariant features and passed into the
      final orbital-shaping layer. Otherwise, just the output of the
      one-electron stream is passed into the orbital-shaping layer.
    hf_solution: If present, initialise the parameters to match the Hartree-Fock
      solution. Otherwise a random initialisation is use.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    determinants: Number of determinants to use.
    after_determinants: currently ignored.

  Returns:
    init, network tuple of callables. init has signature f(key), where key is
    a JAX PRNG state, and returns the pytree of network parameters. network has
    signature network(params, x), where params is the pytree of network
    parameters and x the 3N-dimensional vector of electron positions, and
    returns the network output in log space.
  """

  init = functools.partial(
      init_fermi_net_params,
      atoms=atoms,
      spins=spins,
      envelope_type=envelope_type,
      bias_orbitals=bias_orbitals,
      use_last_layer=use_last_layer,
      hf_solution=hf_solution,
      full_det=full_det,
      hidden_dims=hidden_dims,
      determinants=determinants,
      after_determinants=after_determinants,
  )
  network = functools.partial(
      fermi_net,
      atoms=atoms,
      spins=spins,
      charges=charges,
      envelope_type=envelope_type,
      full_det=full_det)
  return init, network
