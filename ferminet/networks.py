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
import enum
import functools
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import attr
import chex
from ferminet import envelopes
from ferminet import network_blocks
from ferminet import sto
from ferminet.utils import scf
import jax
import jax.numpy as jnp
from typing_extensions import Protocol


FermiLayers = Tuple[Tuple[int, int], ...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]
# pytype: enable=not-supported-yet
# Parameters for a single part of the network are just a dict.
Param = Mapping[str, jnp.ndarray]

## Interfaces (public) ##


class InitFermiNet(Protocol):

  def __call__(self, key: chex.PRNGKey) -> ParamTree:
    """Returns initialized parameters for the network.

    Args:
      key: RNG state
    """


class FermiNetLike(Protocol):

  def __call__(self, params: ParamTree,
               electrons: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the sign and log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
    """


class LogFermiNetLike(Protocol):

  def __call__(self, params: ParamTree, electrons: jnp.ndarray) -> jnp.ndarray:
    """Returns the log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
    """

## Interfaces (network components) ##


class FeatureInit(Protocol):

  def __call__(self) -> Tuple[Tuple[int, int], Param]:
    """Creates the learnable parameters for the feature input layer.

    Returns:
      Tuple of ((x, y), params), where x and y are the number of one-electron
      features per electron and number of two-electron features per pair of
      electrons respectively, and params is a (potentially empty) mapping of
      learnable parameters associated with the feature construction layer.
    """


class FeatureApply(Protocol):

  def __call__(self, ae: jnp.ndarray, r_ae: jnp.ndarray, ee: jnp.ndarray,
               r_ee: jnp.ndarray,
               **params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Creates the features to pass into the network.

    Args:
      ae: electron-atom vectors. Shape: (nelectron, natom, 3).
      r_ae: electron-atom distances. Shape: (nelectron, natom, 1).
      ee: electron-electron vectors. Shape: (nelectron, nelectron, 3).
      r_ee: electron-electron distances. Shape: (nelectron, nelectron).
      **params: learnable parameters, as initialised in the corresponding
        FeatureInit function.
    """


@attr.s(auto_attribs=True)
class FeatureLayer:
  init: FeatureInit
  apply: FeatureApply


class FeatureLayerType(enum.Enum):
  STANDARD = enum.auto()


class MakeFeatureLayer(Protocol):

  def __call__(self,
               charges: jnp.ndarray,
               nspins: Sequence[int],
               ndim: int,
               **kwargs: Any) -> FeatureLayer:
    """Builds the FeatureLayer object.

    Args:
      charges: (natom) array of atom nuclear charges.
      nspins: tuple of the number of spin-up and spin-down electrons.
      ndim: dimension of the system.
      **kwargs: additional kwargs to use for creating the specific FeatureLayer.
    """


## Network settings ##


@attr.s(auto_attribs=True, kw_only=True)
class FermiNetOptions:
  """Options controlling the FermiNet architecture.

  Attributes:
    ndim: dimension of system. Change only with caution.
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    use_last_layer: If true, the outputs of the one- and two-electron streams
      are combined into permutation-equivariant features and passed into the
      final orbital-shaping layer. Otherwise, just the output of the
      one-electron stream is passed into the orbital-shaping layer.
    determinants: Number of determinants to use.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    envelope: Envelope object to create and apply the multiplicative envelope.
    feature_layer: Feature object to create and apply the input features for the
      one- and two-electron layers.
  """
  ndim: int = 3
  hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32), (256, 32))
  use_last_layer: bool = False
  determinants: int = 16
  full_det: bool = True
  bias_orbitals: bool = False
  envelope: envelopes.Envelope = attr.ib(
      default=attr.Factory(
          envelopes.make_isotropic_envelope,
          takes_self=False))
  feature_layer: FeatureLayer = attr.ib(
      default=attr.Factory(
          lambda self: make_ferminet_features(ndim=self.ndim), takes_self=True))


## Network initialisation ##


def init_layers(
    key: chex.PRNGKey, dims_one_in: Sequence[int], dims_one_out: Sequence[int],
    dims_two_in: Sequence[int],
    dims_two_out: Sequence[int]) -> Tuple[Sequence[Param], Sequence[Param]]:
  """Initialises parameters for the FermiNet layers.

  The final two-electron layer is not strictly necessary (i.e.
  FermiNetOptions.use_last_layer is False), in which case the two-electron
  stream contains one fewer layers than the one-electron stream.

  Args:
    key: JAX RNG state.
    dims_one_in: dimension of inputs to each one-electron layer.
    dims_one_out: dimension of outputs (number of hidden units) in each
      one-electron layer.
    dims_two_in: dimension of inputs to each two-electron layer.
    dims_two_out: dimension of outputs (number of hidden units) in each
      two-electron layer.

  Returns:
    Pair of sequences (length: number of layers) of parameters for one- and
    two-electon streams.

  Raises:
    ValueError: if dims_one_in and dims_one_out are different lengths, or
    similarly for dims_two_in and dims_two_out, or if the number of one-electron
    layers is not equal to or one more than the number of two electron layers.
  """
  if len(dims_one_in) != len(dims_one_out):
    raise ValueError(
        'Length of one-electron stream inputs and outputs not identical.')
  if len(dims_two_in) != len(dims_two_out):
    raise ValueError(
        'Length of two-electron stream inputs and outputs not identical.')
  if len(dims_two_in) not in (len(dims_one_out), len(dims_one_out) - 1):
    raise ValueError('Number of layers in two electron stream must match or be '
                     'one fewer than the number of layers in the one-electron '
                     'stream')
  single = []
  double = []
  ndouble_layers = len(dims_two_in)
  for i in range(len(dims_one_in)):
    key, subkey = jax.random.split(key)
    single.append(
        network_blocks.init_linear_layer(
            subkey,
            in_dim=dims_one_in[i],
            out_dim=dims_one_out[i],
            include_bias=True))

    if i < ndouble_layers:
      key, subkey = jax.random.split(key)
      double.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_two_in[i],
              out_dim=dims_two_out[i],
              include_bias=True))

  return single, double


def init_orbital_shaping(
    key: chex.PRNGKey,
    input_dim: int,
    nspin_orbitals: Sequence[int],
    bias_orbitals: bool,
) -> Sequence[Param]:
  """Initialises orbital shaping layer.

  Args:
    key: JAX RNG state.
    input_dim: dimension of input activations to the orbital shaping layer.
    nspin_orbitals: total number of orbitals in each spin-channel.
    bias_orbitals: whether to include a bias in the layer.

  Returns:
    Parameters of length len(nspin_orbitals) for the orbital shaping for each
    spin channel.
  """
  orbitals = []
  for nspin_orbital in nspin_orbitals:
    key, subkey = jax.random.split(key)
    orbitals.append(
        network_blocks.init_linear_layer(
            subkey,
            in_dim=input_dim,
            out_dim=nspin_orbital,
            include_bias=bias_orbitals))
  return orbitals


def init_to_hf_solution(
    hf_solution: scf.Scf,
    single_layers: Sequence[Param],
    orbital_layer: Sequence[Param],
    determinants: int,
    active_spin_channels: Sequence[int],
    eps: float = 0.01) -> Tuple[Sequence[Param], Sequence[Param]]:
  """Sets initial parameters to match Hartree-Fock.

  NOTE: this does not handle the envelope parameters, which are done in the
  appropriate envelope initialisation functions. Not all envelopes support HF
  initialisation.

  Args:
    hf_solution: Hartree-Fock state to match.
    single_layers: parameters (weights and biases) for the one-electron stream,
      with length: number of layers in the one-electron stream.
    orbital_layer: parameters for the orbital-shaping layer, length: number of
      spin-channels in the system.
    determinants: Number of determinants used in the final wavefunction.
    active_spin_channels: Number of particles in each spin channel containing at
      least one particle.
    eps: scaling factor for all weights and biases such that they are
      initialised close to zero unless otherwise required to match Hartree-Fock.

  Returns:
    Tuple of parameters for the one-electron stream and the orbital shaping
    layer respectively.
  """
  # Scale all params in one-electron stream to be near zero.
  single_layers = jax.tree_map(lambda param: param * eps, single_layers)
  # Initialize first layer of Fermi Net to match s- or p-type orbitals.
  # The sto and sto-poly envelopes can exactly capture the s-type orbital,
  # so the effect of the neural network part is constant, while the p-type
  # orbital also has a term multiplied by x, y or z.
  j = 0
  for ia, atom in enumerate(hf_solution.molecule):
    coeffs = sto.STO_6G_COEFFS[atom.symbol]
    for orb in coeffs.keys():
      if orb[1] == 's':
        single_layers[0]['b'] = single_layers[0]['b'].at[j].set(1.0)
        j += 1
      elif orb[1] == 'p':
        w = single_layers[0]['w']
        w = w.at[ia * 4 + 1:(ia + 1) * 4, j:j + 3].set(jnp.eye(3))
        single_layers[0]['w'] = w
        j += 3
      else:
        raise NotImplementedError('HF Initialization not implemented for '
                                  f'{orb[1]} orbitals')
  # Scale all params in orbital shaping to be near zero.
  orbital_layer = jax.tree_map(lambda param: param * eps, orbital_layer)
  for i, spin in enumerate(active_spin_channels):
    # Initialize last layer to match Hartree-Fock weights on basis set.
    norb = hf_solution.mean_field.mo_coeff[i].shape[0]
    mat = hf_solution.mean_field.mo_coeff[i][:, :spin]
    w = orbital_layer[i]['w']
    for j in range(determinants):
      w = w.at[:norb, j * spin:(j + 1) * spin].set(mat)
    orbital_layer[i]['w'] = w
  return single_layers, orbital_layer


def init_fermi_net_params(
    key: chex.PRNGKey,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: FermiNetOptions,
    hf_solution: Optional[scf.Scf] = None,
    eps: float = 0.01,
) -> ParamTree:
  """Initializes parameters for the Fermionic Neural Network.

  Args:
    key: JAX RNG state.
    atoms: (natom, ndim) array of atom positions.
    nspins: A tuple with either the number of spin-up and spin-down electrons,
      or the total number of electrons. If the latter, the spins are instead
      given as an input to the network.
    options: network options.
    hf_solution: If present, initialise the parameters to match the Hartree-Fock
      solution. Otherwise a random initialisation is use.
    eps: If hf_solution is present, scale all weights and biases except the
      first layer by this factor such that they are initialised close to zero.

  Returns:
    PyTree of network parameters. Spin-dependent parameters are only created for
    spin channels containing at least one particle.
  """
  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    if options.bias_orbitals:
      raise ValueError('Cannot bias orbitals w/STO envelope.')
  if hf_solution is not None:
    if options.use_last_layer:
      raise ValueError('Cannot use last layer w/HF init')
    if options.envelope.apply_type not in ('sto', 'sto-poly'):
      raise ValueError('When using HF init, '
                       'envelope_type must be `sto` or `sto-poly`.')

  active_spin_channels = [spin for spin in nspins if spin > 0]
  nchannels = len(active_spin_channels)
  if nchannels == 0:
    raise ValueError('No electrons present!')

  params = {}
  (num_one_features, num_two_features), params['input'] = (
      options.feature_layer.init())

  # The input to layer L of the one-electron stream is from
  # construct_symmetric_features and shape (nelectrons, nfeatures), where
  # nfeatures is i) output from the previous one-electron layer; ii) the mean
  # for each spin channel from each layer; iii) the mean for each spin channel
  # from each two-electron layer. We don't create features for spin channels
  # which contain no electrons (i.e. spin-polarised systems).
  nfeatures = lambda out1, out2: (nchannels + 1) * out1 + nchannels * out2

  natom, ndim = atoms.shape
  # one-electron stream, per electron:
  #  - one-electron features per atom (default: electron-atom vectors
  #    (ndim/atom) and distances (1/atom)),
  # two-electron stream, per pair of electrons:
  #  - two-electron features per electron pair (default: electron-electron
  #    vector (dim) and distance (1))
  feature_one_dims = natom * num_one_features
  feature_two_dims = num_two_features
  dims_one_in = (
      [nfeatures(feature_one_dims, feature_two_dims)] +
      [nfeatures(hdim[0], hdim[1]) for hdim in options.hidden_dims[:-1]])
  dims_one_out = [hdim[0] for hdim in options.hidden_dims]
  if options.use_last_layer:
    dims_two_in = ([feature_two_dims] +
                   [hdim[1] for hdim in options.hidden_dims[:-1]])
    dims_two_out = [hdim[1] for hdim in options.hidden_dims]
  else:
    dims_two_in = ([feature_two_dims] +
                   [hdim[1] for hdim in options.hidden_dims[:-2]])
    dims_two_out = [hdim[1] for hdim in options.hidden_dims[:-1]]

  if not options.use_last_layer:
    # Just pass the activations from the final layer of the one-electron stream
    # directly to orbital shaping.
    dims_orbital_in = options.hidden_dims[-1][0]
  else:
    dims_orbital_in = nfeatures(options.hidden_dims[-1][0],
                                options.hidden_dims[-1][1])

  # How many spin-orbitals do we need to create per spin channel?
  nspin_orbitals = []
  for nspin in active_spin_channels:
    if options.full_det:
      # Dense determinant. Need N orbitals per electron per determinant.
      norbitals = sum(nspins) * options.determinants
    else:
      # Spin-factored block-diagonal determinant. Need nspin orbitals per
      # electron per determinant.
      norbitals = nspin * options.determinants
    nspin_orbitals.append(norbitals)

  # Layer initialisation
  key, subkey = jax.random.split(key, num=2)
  params['single'], params['double'] = init_layers(
      key=subkey,
      dims_one_in=dims_one_in,
      dims_one_out=dims_one_out,
      dims_two_in=dims_two_in,
      dims_two_out=dims_two_out)

  # create envelope params
  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    # Applied to output from final layer of 1e stream.
    output_dims = dims_orbital_in
  elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    # Applied to orbitals.
    output_dims = nspin_orbitals
  elif options.envelope.apply_type == envelopes.EnvelopeType.POST_DETERMINANT:
    # Applied to all determinants.
    output_dims = 1
  else:
    raise ValueError('Unknown envelope type')
  params['envelope'] = options.envelope.init(
      natom=natom, output_dims=output_dims, hf=hf_solution, ndim=ndim)

  # orbital shaping
  key, subkey = jax.random.split(key, num=2)
  params['orbital'] = init_orbital_shaping(
      key=subkey,
      input_dim=dims_orbital_in,
      nspin_orbitals=nspin_orbitals,
      bias_orbitals=options.bias_orbitals)

  if hf_solution is not None:
    params['single'], params['orbital'] = init_to_hf_solution(
        hf_solution=hf_solution,
        single_layers=params['single'],
        orbital_layer=params['orbital'],
        determinants=options.determinants,
        active_spin_channels=active_spin_channels,
        eps=eps)

  return params

## Network layers ##


def construct_input_features(
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Constructs inputs to Fermi Net from raw electron and atomic positions.

  Args:
    pos: electron positions. Shape (nelectrons*ndim,).
    atoms: atom positions. Shape (natoms, ndim).
    ndim: dimension of system. Change only with caution.

  Returns:
    ae, ee, r_ae, r_ee tuple, where:
      ae: atom-electron vector. Shape (nelectron, natom, ndim).
      ee: atom-electron vector. Shape (nelectron, nelectron, ndim).
      r_ae: atom-electron distance. Shape (nelectron, natom, 1).
      r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
    The diagonal terms in r_ee are masked out such that the gradients of these
    terms are also zero.
  """
  assert atoms.shape[1] == ndim
  ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
  ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])

  r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
  # Avoid computing the norm of zero, as is has undefined grad
  n = ee.shape[0]
  r_ee = (
      jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))

  return ae, ee, r_ae, r_ee[..., None]


def make_ferminet_features(charges: Optional[jnp.ndarray] = None,
                           nspins: Optional[Tuple[int, ...]] = None,
                           ndim: int = 3) -> FeatureLayer:
  """Returns the init and apply functions for the standard features."""

  del charges, nspins

  def init() -> Tuple[Tuple[int, int], Param]:
    return (ndim + 1, ndim + 1), {}

  def apply(ae, r_ae, ee, r_ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ae_features = jnp.concatenate((r_ae, ae), axis=2)
    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
    ee_features = jnp.concatenate((r_ee, ee), axis=2)
    return ae_features, ee_features

  return FeatureLayer(init=init, apply=apply)


def construct_symmetric_features(h_one: jnp.ndarray, h_two: jnp.ndarray,
                                 nspins: Tuple[int, int]) -> jnp.ndarray:
  """Combines intermediate features from rank-one and -two streams.

  Args:
    h_one: set of one-electron features. Shape: (nelectrons, n1), where n1 is
      the output size of the previous layer.
    h_two: set of two-electron features. Shape: (nelectrons, nelectrons, n2),
      where n2 is the output size of the previous layer.
    nspins: Number of spin-up and spin-down electrons.

  Returns:
    array containing the permutation-equivariant features: the input set of
    one-electron features, the mean of the one-electron features over each
    (occupied) spin channel, and the mean of the two-electron features over each
    (occupied) spin channel. Output shape (nelectrons, 3*n1 + 2*n2) if there are
    both spin-up and spin-down electrons and (nelectrons, 2*n1 + n2) otherwise.
  """
  # Split features into spin up and spin down electrons
  spin_partitions = network_blocks.array_partitions(nspins)
  h_ones = jnp.split(h_one, spin_partitions, axis=0)
  h_twos = jnp.split(h_two, spin_partitions, axis=0)

  # Construct inputs to next layer
  # h.size == 0 corresponds to unoccupied spin channels.
  g_one = [jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0]
  g_two = [jnp.mean(h, axis=0) for h in h_twos if h.size > 0]

  g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]

  return jnp.concatenate([h_one] + g_one + g_two, axis=1)


def fermi_net_orbitals(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: FermiNetOptions = FermiNetOptions(),
):
  """Forward evaluation of the Fermionic Neural Network up to the orbitals.

  Args:
    params: A dictionary of parameters, containing fields:
      `atoms`: atomic positions, used to construct input features.
      `single`: a list of dictionaries with params 'w' and 'b', weights for the
        one-electron stream of the network.
      `double`: a list of dictionaries with params 'w' and 'b', weights for the
        two-electron stream of the network.
      `orbital`: a list of two weight matrices, for spin up and spin down (no
        bias is necessary as it only adds a constant to each row, which does not
        change the determinant).
      `dets`: weight on the linear combination of determinants
      `envelope`: a dictionary with fields `sigma` and `pi`, weights for the
        multiplicative envelope.
    pos: The electron positions, a 3N dimensional vector.
    atoms: Array with positions of atoms.
    nspins: Tuple with number of spin up and spin down electrons.
    options: Network configuration.

  Returns:
    One matrix (two matrices if options.full_det is False) that exchange columns
    under the exchange of inputs of shape (ndet, nalpha+nbeta, nalpha+nbeta) (or
    (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)) and a tuple of (ae, r_ae,
    r_ee), the atom-electron vectors, distances and electron-electron distances.
  """

  ae, ee, r_ae, r_ee = construct_input_features(pos, atoms)
  ae_features, ee_features = options.feature_layer.apply(
      ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input'])

  h_one = ae_features  # single-electron features
  h_two = ee_features  # two-electron features
  residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
  for i in range(len(params['double'])):
    h_one_in = construct_symmetric_features(h_one, h_two, nspins)

    # Execute next layer
    h_one_next = jnp.tanh(
        network_blocks.linear_layer(h_one_in, **params['single'][i]))
    h_two_next = jnp.tanh(
        network_blocks.vmap_linear_layer(h_two, params['double'][i]['w'],
                                         params['double'][i]['b']))
    h_one = residual(h_one, h_one_next)
    h_two = residual(h_two, h_two_next)
  if len(params['double']) != len(params['single']):
    h_one_in = construct_symmetric_features(h_one, h_two, nspins)
    h_one_next = jnp.tanh(
        network_blocks.linear_layer(h_one_in, **params['single'][-1]))
    h_one = residual(h_one, h_one_next)
    h_to_orbitals = h_one
  else:
    h_to_orbitals = construct_symmetric_features(h_one, h_two, nspins)
  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
    envelope_factor = options.envelope.apply(
        ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope'])
    h_to_orbitals = envelope_factor * h_to_orbitals
  # Note split creates arrays of size 0 for spin channels without any electrons.
  h_to_orbitals = jnp.split(
      h_to_orbitals, network_blocks.array_partitions(nspins), axis=0)
  # Drop unoccupied spin channels
  h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
  active_spin_channels = [spin for spin in nspins if spin > 0]
  active_spin_partitions = network_blocks.array_partitions(active_spin_channels)
  # Create orbitals.
  orbitals = [
      network_blocks.linear_layer(h, **p)
      for h, p in zip(h_to_orbitals, params['orbital'])
  ]

  # Apply envelopes if required.
  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
    r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
    r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
    for i in range(len(active_spin_channels)):
      orbitals[i] = orbitals[i] * options.envelope.apply(
          ae=ae_channels[i],
          r_ae=r_ae_channels[i],
          r_ee=r_ee_channels[i],
          **params['envelope'][i],
      )

  # Reshape into matrices.
  shapes = [(spin, -1, sum(nspins) if options.full_det else spin)
            for spin in active_spin_channels]
  orbitals = [
      jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
  ]
  orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
  if options.full_det:
    orbitals = [jnp.concatenate(orbitals, axis=1)]
  return orbitals, (ae, r_ae, r_ee)

## FermiNet ##


def fermi_net(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: FermiNetOptions = FermiNetOptions(),
):
  """Forward evaluation of the Fermionic Neural Network for a single datum.

  Args:
    params: A dictionary of parameters, containing fields:
      `atoms`: atomic positions, used to construct input features.
      `single`: a list of dictionaries with params 'w' and 'b', weights for the
        one-electron stream of the network.
      `double`: a list of dictionaries with params 'w' and 'b', weights for the
        two-electron stream of the network.
      `orbital`: a list of two weight matrices, for spin up and spin down (no
        bias is necessary as it only adds a constant to each row, which does not
        change the determinant).
      `dets`: weight on the linear combination of determinants
      `envelope`: a dictionary with fields `sigma` and `pi`, weights for the
        multiplicative envelope.
    pos: The electron positions, a 3N dimensional vector.
    atoms: Array with positions of atoms.
    nspins: Tuple with number of spin up and spin down electrons.
    options: network options.

  Returns:
    Output of antisymmetric neural network in log space, i.e. a tuple of sign of
    and log absolute of the network evaluated at x.
  """

  orbitals, (ae, r_ae, r_ee) = fermi_net_orbitals(
      params,
      pos,
      atoms=atoms,
      nspins=nspins,
      options=options,
  )
  output = network_blocks.logdet_matmul(orbitals)
  if options.envelope.apply_type == envelopes.EnvelopeType.POST_DETERMINANT:
    output = output[0], output[1] + options.envelope.apply(
        ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope'])
  return output


def make_fermi_net(
    atoms: jnp.ndarray,
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    *,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[FeatureLayer] = None,
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    hf_solution: Optional[scf.Scf] = None,
    full_det: bool = True,
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    after_determinants: Union[int, Tuple[int, ...]] = 1,
) -> Tuple[InitFermiNet, FermiNetLike, FermiNetOptions]:
  """Creates functions for initializing parameters and evaluating ferminet.

  Args:
    atoms: (natom, ndim) array of atom positions.
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
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
    init, network, options tuple, where init and network are callables which
    initialise the network parameters and apply the network respectively, and
    options specifies the settings used in the network.
  """
  del after_determinants

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    feature_layer = make_ferminet_features(charges, nspins)

  options = FermiNetOptions(
      hidden_dims=hidden_dims,
      use_last_layer=use_last_layer,
      determinants=determinants,
      full_det=full_det,
      bias_orbitals=bias_orbitals,
      envelope=envelope,
      feature_layer=feature_layer,
  )

  init = functools.partial(
      init_fermi_net_params,
      atoms=atoms,
      nspins=nspins,
      options=options,
      hf_solution=hf_solution,
  )
  network = functools.partial(
      fermi_net,
      atoms=atoms,
      nspins=nspins,
      options=options,
  )

  return init, network, options
