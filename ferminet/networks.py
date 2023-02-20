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
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import attr
import chex
from ferminet import envelopes
from ferminet import network_blocks
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


@chex.dataclass
class FermiNetData:
  """Data passed to network.

  Shapes given for an unbatched element (i.e. a single MCMC configuration).

  NOTE:
    the networks are written in batchless form. Typically one then maps
    (pmap+vmap) over every attribute of FermiNetData (nb this is required if
    using KFAC, as it assumes the FIM is estimated over a batch of data), but
    this is not strictly required. If some attributes are not mapped, then JAX
    simply broadcasts them to the mapped dimensions (i.e. those attributes are
    treated as identical for every MCMC configuration.

  Attributes:
    positions: walker positions, shape (nelectrons*ndim).
    spins: spins of each walker, shape (nelectrons).
    atoms: atomic positions, shape (natoms*ndim).
    charges: atomic charges, shape (natoms).
  """

  # We need to be able to construct instances of this with leaf nodes as jax
  # arrays (for the actual data) and as integers (to use with in_axes for
  # jax.vmap etc). We can type the struct to be either all arrays or all ints
  # using Generic, it just slightly complicates the type annotations in a few
  # functions (i.e. requires FermiNetData[jnp.ndarray] annotation).
  positions: Any
  spins: Any
  atoms: Any
  charges: Any


## Interfaces (public) ##


class InitFermiNet(Protocol):

  def __call__(self, key: chex.PRNGKey) -> ParamTree:
    """Returns initialized parameters for the network.

    Args:
      key: RNG state
    """


class FermiNetLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      electrons: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the sign and log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
      spins: 1D array specifying the spin state of each electron.
      atoms: positions of nuclei, shape: (natoms, ndim).
      charges: nuclei charges, shape: (natoms).
    """


class LogFermiNetLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      electrons: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
      spins: 1D array specifying the spin state of each electron.
      atoms: positions of nuclei, shape: (natoms, ndim).
      charges: nuclear charges, shape: (natoms).
    """


class OrbitalFnLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      Sequence of orbitals.
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

  def __call__(
      self,
      natoms: int,
      nspins: Sequence[int],
      ndim: int,
      **kwargs: Any,
  ) -> FeatureLayer:
    """Builds the FeatureLayer object.

    Args:
      natoms: number of atoms.
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
  feature_layer: FeatureLayer = None


# Network class.


@attr.s(auto_attribs=True)
class Network:
  options: FermiNetOptions
  init: InitFermiNet
  apply: FermiNetLike
  orbitals: OrbitalFnLike


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
    two-electron streams.

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


def make_ferminet_features(
    natoms: int, nspins: Optional[Tuple[int, ...]] = None, ndim: int = 3
) -> FeatureLayer:
  """Returns the init and apply functions for the standard features."""

  del nspins

  def init() -> Tuple[Tuple[int, int], Param]:
    return (natoms * (ndim + 1), ndim + 1), {}

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


def make_orbitals(
    nspins: Tuple[int, ...], charges: jnp.ndarray, options: FermiNetOptions
) -> ...:
  """Returns init, apply pair for orbitals.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    charges: (atom) array of atomic nuclear charges.
    options: Network configuration.
  """

  def init(key: chex.PRNGKey) -> ParamTree:
    """Returns initial random parameters for creating orbitals.

    Args:
      key: RNG state.
    """
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      if options.bias_orbitals:
        raise ValueError('Cannot bias orbitals w/STO envelope.')

    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)
    if nchannels == 0:
      raise ValueError('No electrons present!')

    params = {}
    (num_one_features, num_two_features), params['input'] = (
        options.feature_layer.init()
    )

    # The input to layer L of the one-electron stream is from
    # construct_symmetric_features and shape (nelectrons, nfeatures), where
    # nfeatures is i) output from the previous one-electron layer; ii) the mean
    # for each spin channel from each layer; iii) the mean for each spin channel
    # from each two-electron layer. We don't create features for spin channels
    # which contain no electrons (i.e. spin-polarised systems).
    nfeatures = lambda out1, out2: (nchannels + 1) * out1 + nchannels * out2

    natom = charges.shape[0]
    # one-electron stream, per electron:
    #  - one-electron features per atom (default: electron-atom vectors
    #    (ndim/atom) and distances (1/atom)),
    # two-electron stream, per pair of electrons:
    #  - two-electron features per electron pair (default: electron-electron
    #    vector (dim) and distance (1))
    feature_one_dims = num_one_features
    feature_two_dims = num_two_features
    dims_one_in = [nfeatures(feature_one_dims, feature_two_dims)] + [
        nfeatures(hdim[0], hdim[1]) for hdim in options.hidden_dims[:-1]
    ]
    dims_one_out = [hdim[0] for hdim in options.hidden_dims]
    if options.use_last_layer:
      dims_two_in = [feature_two_dims] + [
          hdim[1] for hdim in options.hidden_dims[:-1]
      ]
      dims_two_out = [hdim[1] for hdim in options.hidden_dims]
    else:
      dims_two_in = [feature_two_dims] + [
          hdim[1] for hdim in options.hidden_dims[:-2]
      ]
      dims_two_out = [hdim[1] for hdim in options.hidden_dims[:-1]]

    if not options.use_last_layer:
      # Just pass the activations from the final layer of the one-electron
      # stream directly to orbital shaping.
      dims_orbital_in = options.hidden_dims[-1][0]
    else:
      dims_orbital_in = nfeatures(
          options.hidden_dims[-1][0], options.hidden_dims[-1][1]
      )

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
        dims_two_out=dims_two_out,
    )

    # create envelope params
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      # Applied to output from final layer of 1e stream.
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # Applied to orbitals.
      output_dims = nspin_orbitals
    else:
      raise ValueError('Unknown envelope type')
    params['envelope'] = options.envelope.init(
        natom=natom, output_dims=output_dims, ndim=options.ndim
    )

    # orbital shaping
    key, subkey = jax.random.split(key, num=2)
    params['orbital'] = init_orbital_shaping(
        key=subkey,
        input_dim=dims_orbital_in,
        nspin_orbitals=nspin_orbitals,
        bias_orbitals=options.bias_orbitals,
    )

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      One matrix (two matrices if options.full_det is False) that exchange
      columns under the exchange of inputs of shape (ndet, nalpha+nbeta,
      nalpha+nbeta) (or (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)).
    """
    del spins, charges

    ae, ee, r_ae, r_ee = construct_input_features(pos, atoms, ndim=options.ndim)
    ae_features, ee_features = options.feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
    )

    h_one = ae_features  # single-electron features
    h_two = ee_features  # two-electron features
    residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y
    for i in range(len(params['double'])):
      h_one_in = construct_symmetric_features(h_one, h_two, nspins)

      # Execute next layer
      h_one_next = jnp.tanh(
          network_blocks.linear_layer(h_one_in, **params['single'][i])
      )
      h_two_next = jnp.tanh(
          network_blocks.vmap_linear_layer(
              h_two, params['double'][i]['w'], params['double'][i]['b']
          )
      )
      h_one = residual(h_one, h_one_next)
      h_two = residual(h_two, h_two_next)
    if len(params['double']) != len(params['single']):
      h_one_in = construct_symmetric_features(h_one, h_two, nspins)
      h_one_next = jnp.tanh(
          network_blocks.linear_layer(h_one_in, **params['single'][-1])
      )
      h_one = residual(h_one, h_one_next)
      h_to_orbitals = h_one
    else:
      h_to_orbitals = construct_symmetric_features(h_one, h_two, nspins)
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      envelope_factor = options.envelope.apply(
          ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
      )
      h_to_orbitals = envelope_factor * h_to_orbitals
    # Note split creates arrays of size 0 for spin channels without electrons.
    h_to_orbitals = jnp.split(
        h_to_orbitals, network_blocks.array_partitions(nspins), axis=0
    )
    # Drop unoccupied spin channels
    h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]
    active_spin_channels = [spin for spin in nspins if spin > 0]
    active_spin_partitions = network_blocks.array_partitions(
        active_spin_channels
    )
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
    shapes = [
        (spin, -1, sum(nspins) if options.full_det else spin)
        for spin in active_spin_channels
    ]
    orbitals = [
        jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)
    ]
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    if options.full_det:
      orbitals = [jnp.concatenate(orbitals, axis=1)]
    return orbitals

  return init, apply


## FermiNet ##


def make_fermi_net(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    *,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[FeatureLayer] = None,
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    full_det: bool = True,
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    ndim: int = 3,
) -> Network:
  """Creates functions for initializing parameters and evaluating ferminet.

  Args:
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
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    determinants: Number of determinants to use.
    ndim: dimension of the system.

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """
  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = make_ferminet_features(natoms, nspins, ndim=ndim)

  options = FermiNetOptions(
      ndim=ndim,
      hidden_dims=hidden_dims,
      use_last_layer=use_last_layer,
      determinants=determinants,
      full_det=full_det,
      bias_orbitals=bias_orbitals,
      envelope=envelope,
      feature_layer=feature_layer,
  )

  orbitals_init, orbitals_apply = make_orbitals(
      nspins=nspins, charges=charges, options=options
  )

  def init(key: chex.PRNGKey) -> ParamTree:
    key, subkey = jax.random.split(key, num=2)
    return orbitals_init(subkey)

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network for a single datum.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute of the network evaluated at x.
    """

    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    return network_blocks.logdet_matmul(orbitals)

  return Network(
      options=options, init=init, apply=apply, orbitals=orbitals_apply
  )
