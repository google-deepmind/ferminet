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
# limitations under the License

"""Ewald summation of Coulomb Hamiltonian in periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

from typing import Sequence, Tuple

import chex
from ferminet import hamiltonian
from ferminet import networks
from ferminet import pseudopotential as pp
from ferminet.pbc import ewald
import jax.numpy as jnp


def _reduce_to_cell(
    x: jnp.ndarray, lattice: jnp.ndarray, rec: jnp.ndarray
) -> jnp.ndarray:
  """Reduces displacements into the primitive cell centered at the origin."""
  phase = jnp.einsum('il,...l->...i', rec / (2 * jnp.pi), x)
  return jnp.einsum('il,...l->...i', lattice, phase - jnp.round(phase))


def _move_to_mic(
    pos: jnp.ndarray,
    ae: jnp.ndarray,
    lattice: jnp.ndarray,
    rec: jnp.ndarray,
) -> jnp.ndarray:
  """Translates electrons by lattice vectors to the MIC of the closest atom."""
  prim_ae = _reduce_to_cell(ae, lattice, rec)
  closest = jnp.argmin(jnp.linalg.norm(prim_ae, axis=-1), axis=1)
  idx = jnp.arange(ae.shape[0])
  shift = prim_ae[idx, closest] - ae[idx, closest]
  return pos + jnp.reshape(shift, pos.shape)


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    ndim: int = 3,
    complex_output: bool = False,
    laplacian_method: str = 'default',
    states: int = 0,
    state_specific: bool = False,
    pp_type: str = 'ccecp',
    pp_symbols: Sequence[str] | None = None,
    lattice: jnp.ndarray | None = None,
    convergence_radius: int = 5,
) -> hamiltonian.LocalEnergy:
  """Creates the local energy function in periodic boundary conditions.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    ndim: Number of dimensions.
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of: 'default': take
      jvp(grad), looping over inputs 'folx': use Microsoft's implementation of
      forward laplacian
    states: Number of excited states to compute. Not implemented, only present
      for consistency of calling convention.
    state_specific: Not implemented.
    pp_type: type of pseudopotential to use. Only used if pp_symbols is
      provided.
    pp_symbols: sequence of element symbols for which the pseudopotential is
      used.
    lattice: Shape (ndim, ndim). Matrix of lattice vectors. Default: identity
      matrix.
    convergence_radius: int. Radius of cluster summed over by Ewald sums.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  if states > 0 or state_specific:
    raise NotImplementedError('Excited states not implemented with PBC.')

  del nspins
  if lattice is None:
    lattice = jnp.eye(ndim)

  if not pp_symbols:
    effective_charges = charges
    use_pp = False
  elif ndim == 3:
    effective_charges, pp_local, pp_nonlocal = pp.make_pp_potential(
        charges=charges,
        symbols=pp_symbols,
        quad_degree=4,
        ecp=pp_type,
        complex_output=complex_output,
    )
    use_pp = not bool(jnp.all(effective_charges == charges))
  else:
    raise NotImplementedError('Pseudopotentials only implemented for 3D.')

  ke = hamiltonian.local_kinetic_energy(f,
                                        use_scan=use_scan,
                                        complex_output=complex_output,
                                        laplacian_method=laplacian_method)
  rec = 2 * jnp.pi * jnp.linalg.inv(lattice)

  def _e_l(
      params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
  ) -> Tuple[jnp.ndarray, jnp.ndarray | None]:
    """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    potential_energy = ewald.make_ewald_potential(
        lattice=lattice,
        atoms=data.atoms,
        charges=effective_charges,
        truncation_limit=convergence_radius,
        ndim=ndim,
    )
    ae, ee, _, _ = networks.construct_input_features(
        data.positions, data.atoms, ndim
    )
    if use_pp:
      pos = _move_to_mic(data.positions, ae, lattice, rec)
      ae, ee, _, _ = networks.construct_input_features(pos, data.atoms, ndim)
      ae = _reduce_to_cell(ae, lattice, rec)
      ee = _reduce_to_cell(ee, lattice, rec)
      r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
      data = networks.FermiNetData(
          positions=pos,
          spins=data.spins,
          atoms=data.atoms,
          charges=data.charges,
      )
      potential = (
          potential_energy(ae, ee)
          + pp_local(r_ae)
          + pp_nonlocal(key, f, params, data, ae, r_ae)
      )
    else:
      del key  # unused
      potential = potential_energy(ae, ee)
    kinetic = ke(params, data)
    return potential + kinetic, None

  return _e_l
