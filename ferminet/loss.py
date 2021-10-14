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

"""Helper functions to create the loss and custom gradient of the loss."""

import chex
from ferminet import constants
from ferminet import hamiltonian
import jax
import jax.numpy as jnp

from kfac_ferminet_alpha import loss_functions


@chex.dataclass
class AuxiliaryLossData:
  """Auxiliary data returned by total_energy.

  Attributes:
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
  """
  variance: jnp.DeviceArray
  local_energy: jnp.DeviceArray


def make_loss(network, batch_network, atoms, charges, clip_local_energy=0.0):
  """Creates the loss function, including custom gradients.

  Args:
    network: function, signature (params, data), which evaluates the log of the
      magnitude of the wavefunction (square root of the log probability
      distribution) at the single MCMC configuration in data given the network
      parameters.
    batch_network: as for network but data is a batch of MCMC configurations.
    atoms: array of (natoms, ndim) specifying the positions of the nuclei.
    charges: array of (natoms) specifying the nuclear charges.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.

  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  el_fun = hamiltonian.local_energy(network, atoms, charges)
  batch_local_energy = jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)

  @jax.custom_jvp
  def total_energy(params, data):
    """Evaluates the total energy of the network for a batch of configurations.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    e_l = batch_local_energy(params, data)
    loss = constants.pmean(jnp.mean(e_l))
    variance = constants.pmean(jnp.mean((e_l - loss)**2))
    return loss, AuxiliaryLossData(variance=variance, local_energy=e_l)

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params, data = primals
    loss, aux_data = total_energy(params, data)

    if clip_local_energy > 0.0:
      # Try centering the window around the median instead of the mean?
      tv = jnp.mean(jnp.abs(aux_data.local_energy - loss))
      tv = constants.pmean(tv)
      diff = jnp.clip(aux_data.local_energy, loss - clip_local_energy * tv,
                      loss + clip_local_energy * tv) - loss
    else:
      diff = aux_data.local_energy - loss

    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    loss_functions.register_normal_predictive_distribution(psi_primal[:, None])
    primals_out = loss, aux_data
    tangents_out = (jnp.dot(psi_tangent, diff), aux_data)
    return primals_out, tangents_out

  return total_energy
