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

from typing import Callable, Sequence, Tuple

import chex
from ferminet import hamiltonian
from ferminet import networks
from ferminet.pbc.ewald import make_ewald_potential
import jax.numpy as jnp


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
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian
    states: Number of excited states to compute. Not implemented, only present
      for consistency of calling convention.
    state_specific: Not implemented.
    pp_type: type of pseudopotential to use. Not implemented.
    pp_symbols: sequence of element symbols for which the pseudopotential is
      used. Not implemented.
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
  if pp_symbols:
    raise NotImplementedError('Pseudopotentials not implemented with PBC.')

  del nspins
  del pp_type

  ke = hamiltonian.local_kinetic_energy(f,
                                        use_scan=use_scan,
                                        complex_output=complex_output,
                                        laplacian_method=laplacian_method)
  
  def _e_l(
      params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
  ) -> Tuple[jnp.ndarray, jnp.ndarray | None]:
    """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    del key  # unused
    potential_energy = make_ewald_potential(
        lattice, data.atoms, charges, convergence_radius, ndim
    )
    ae, ee, _, _ = networks.construct_input_features(
        data.positions, data.atoms, ndim)
    potential = potential_energy(ae, ee)
    kinetic = ke(params, data)
    return potential + kinetic, None

  return _e_l
