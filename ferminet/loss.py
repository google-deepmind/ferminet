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

import functools
from typing import Tuple

import chex
from ferminet import constants
from ferminet import hamiltonian
from ferminet import networks
import folx
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol


@chex.dataclass
class AuxiliaryLossData:
  """Auxiliary data returned by total_energy.

  Attributes:
    energy: mean energy over batch, and over all devices if inside a pmap.
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
    clipped_energy: local energy after clipping has been applied
    grad_local_energy: gradient of the local energy.
    local_energy_mat: for excited states, the local energy matrix.
    s_ij: Matrix of overlaps between wavefunctions.
    mean_s_ij: Mean value of the overlap between wavefunctions across walkers.
  """
  energy: jax.Array  # for some losses, the energy and loss are not the same
  variance: jax.Array
  local_energy: jax.Array
  clipped_energy: jax.Array
  grad_local_energy: jax.Array | None = None
  local_energy_mat: jax.Array | None = None
  s_ij: jax.Array | None = None
  mean_s_ij: jax.Array | None = None


class LossFn(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched data elements to pass to the network.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """


def clip_local_values(
    local_values: jnp.ndarray,
    mean_local_values: jnp.ndarray,
    clip_scale: float,
    clip_from_median: bool,
    center_at_clipped_value: bool,
    complex_output: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Clips local operator estimates to remove outliers.

  Args:
    local_values: batch of local values,  Of/f, where f is the wavefunction and
      O is the operator of interest.
    mean_local_values: mean (over the global batch) of the local values.
    clip_scale: clip local quantities that are outside nD of the estimate of the
      expectation value of the operator, where n is this value and D the mean
      absolute deviation of the local quantities from the estimate of w, to the
      boundaries. The clipped local quantities should only be used to evaluate
      gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate/robust to outliers.
    center_at_clipped_value: If true, center the local energy differences passed
      back to the gradient around the clipped quantities, so the mean difference
      across the batch is guaranteed to be zero.
    complex_output: If true, the local energies will be complex valued.

  Returns:
    Tuple of the central value (estimate of the expectation value of the
    operator) and deviations from the central value for each element in the
    batch. If per_device_threshold is True, then the central value is per
    device.
  """

  batch_mean = lambda values: constants.pmean(jnp.mean(values, axis=0))

  def clip_at_total_variation(values, center, scale):
    tv = batch_mean(jnp.abs(values- center))
    return jnp.clip(values, center - scale * tv, center + scale * tv)

  if clip_from_median:
    # More natural place to center the clipping, but expensive due to both
    # the median and all_gather (at least on multihost)
    all_local_values = constants.all_gather(local_values).real
    shape = all_local_values.shape
    if all_local_values.ndim == 3:  # energy_and_overlap case
      all_local_values = all_local_values.reshape([-1, shape[-1]])
    else:
      all_local_values = all_local_values.reshape([-1])
    clip_center = jnp.median(all_local_values, axis=0)
  else:
    clip_center = mean_local_values
  # roughly, the total variation of the local energies
  if complex_output:
    clipped_local_values = (
        clip_at_total_variation(
            local_values.real, clip_center.real, clip_scale) +
        1.j * clip_at_total_variation(
            local_values.imag, clip_center.imag, clip_scale)
    )
  else:
    clipped_local_values = clip_at_total_variation(
        local_values, clip_center, clip_scale)
  if center_at_clipped_value:
    diff_center = batch_mean(clipped_local_values)
  else:
    diff_center = mean_local_values
  diff = clipped_local_values - diff_center
  return diff_center, diff


def make_loss(network: networks.LogFermiNetLike,
              local_energy: hamiltonian.LocalEnergy,
              clip_local_energy: float = 0.0,
              clip_from_median: bool = True,
              center_at_clipped_energy: bool = True,
              complex_output: bool = False,
              max_vmap_batch_size: int = 0) -> LossFn:
  """Creates the loss function, including custom gradients.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a
      single MCMC configuration given the network parameters.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate.
    center_at_clipped_energy: If true, center the local energy differences
      passed back to the gradient around the clipped local energy, so the mean
      difference across the batch is guaranteed to be zero.
    complex_output: If true, the local energies will be complex valued.
    max_vmap_batch_size: If 0, use standard vmap. If >0, use batched_vmap with
      the given batch size.

  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  vmap = jax.vmap if max_vmap_batch_size == 0 else functools.partial(
      folx.batched_vmap, max_batch_size=max_vmap_batch_size)
  batch_local_energy = vmap(
      local_energy,
      in_axes=(
          None,
          0,
          networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0),
      ),
      out_axes=(0, 0)
  )
  batch_network = vmap(network, in_axes=(None, 0, 0, 0, 0), out_axes=0)

  @jax.custom_jvp
  def total_energy(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    keys = jax.random.split(key, num=data.positions.shape[0])
    e_l, e_l_mat = batch_local_energy(params, keys, data)
    loss = constants.pmean(jnp.mean(e_l))
    loss_diff = e_l - loss
    variance = constants.pmean(jnp.mean(loss_diff * jnp.conj(loss_diff)))
    return loss, AuxiliaryLossData(
        energy=loss,
        variance=variance.real,
        local_energy=e_l,
        clipped_energy=e_l,
        local_energy_mat=e_l_mat,
    )

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params, key, data = primals
    loss, aux_data = total_energy(params, key, data)

    if clip_local_energy > 0.0:
      aux_data.clipped_energy, diff = clip_local_values(
          aux_data.local_energy,
          loss,
          clip_local_energy,
          clip_from_median,
          center_at_clipped_energy,
          complex_output)
    else:
      diff = aux_data.local_energy - loss

    # Due to the simultaneous requirements of KFAC (calling convention must be
    # (params, rng, data)) and Laplacian calculation (only want to take
    # Laplacian wrt electron positions) we need to change up the calling
    # convention between total_energy and batch_network
    data = primals[2]
    data_tangents = tangents[2]
    primals = (primals[0], data.positions, data.spins, data.atoms, data.charges)
    tangents = (
        tangents[0],
        data_tangents.positions,
        data_tangents.spins,
        data_tangents.atoms,
        data_tangents.charges,
    )
    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    if complex_output:
      clipped_el = diff + aux_data.clipped_energy
      term1 = (jnp.dot(clipped_el, jnp.conjugate(psi_tangent)) +
               jnp.dot(jnp.conjugate(clipped_el), psi_tangent))
      term2 = jnp.sum(aux_data.clipped_energy*psi_tangent.real)
      kfac_jax.register_normal_predictive_distribution(
          psi_primal.real[:, None])
      primals_out = loss.real, aux_data
      device_batch_size = jnp.shape(aux_data.local_energy)[0]
      tangents_out = ((term1 - 2*term2).real / device_batch_size, aux_data)
    else:
      kfac_jax.register_normal_predictive_distribution(psi_primal[:, None])
      primals_out = loss, aux_data
      device_batch_size = jnp.shape(aux_data.local_energy)[0]
      tangents_out = (jnp.dot(psi_tangent, diff) / device_batch_size, aux_data)
    return primals_out, tangents_out

  return total_energy


def make_wqmc_loss(
    network: networks.LogFermiNetLike,
    local_energy: hamiltonian.LocalEnergy,
    clip_local_energy: float = 0.0,
    clip_from_median: bool = True,
    center_at_clipped_energy: bool = True,
    complex_output: bool = False,
    max_vmap_batch_size: int = 0,
    vmc_weight: float = 0.0
) -> LossFn:
  """Creates the WQMC loss function, including custom gradients.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a single
      MCMC configuration given the network parameters.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate.
    center_at_clipped_energy: If true, center the local energy differences
      passed back to the gradient around the clipped local energy, so the mean
      difference across the batch is guaranteed to be zero.
    complex_output: If true, the local energies will be complex valued.
    max_vmap_batch_size: If 0, use standard vmap. If >0, use batched_vmap with
      the given batch size.
    vmc_weight: The weight of the contribution from the standard VMC energy
      gradient.

  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  vmap = jax.vmap if max_vmap_batch_size == 0 else functools.partial(
      folx.batched_vmap, max_batch_size=max_vmap_batch_size)
  batch_local_energy = vmap(
      local_energy,
      in_axes=(
          None,
          0,
          networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0),
      ),
      out_axes=(0, 0)
  )
  batch_network = vmap(network, in_axes=(None, 0, 0, 0, 0), out_axes=0)

  @jax.custom_jvp
  def total_energy(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    keys = jax.random.split(key, num=data.positions.shape[0])
    e_l, e_l_mat = batch_local_energy(params, keys, data)
    loss = constants.pmean(jnp.mean(e_l))
    loss_diff = e_l - loss
    variance = constants.pmean(jnp.mean(loss_diff * jnp.conj(loss_diff)))

    def batch_local_energy_pos(pos):
      network_data = networks.FermiNetData(
          positions=pos,
          spins=data.spins,
          atoms=data.atoms,
          charges=data.charges,
      )
      return batch_local_energy(params, keys, network_data)[0].sum()

    grad_e_l = jax.grad(batch_local_energy_pos)(data.positions)
    grad_e_l = jnp.tanh(jax.lax.stop_gradient(grad_e_l))
    return loss, AuxiliaryLossData(
        energy=loss,
        variance=variance.real,
        local_energy=e_l,
        clipped_energy=e_l,
        grad_local_energy=grad_e_l,
        local_energy_mat=e_l_mat,
    )

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params, key, data = primals
    loss, aux_data = total_energy(params, key, data)

    if clip_local_energy > 0.0:
      aux_data.clipped_energy, diff = clip_local_values(
          aux_data.local_energy,
          loss,
          clip_local_energy,
          clip_from_median,
          center_at_clipped_energy,
          complex_output,
      )
    else:
      diff = aux_data.local_energy - loss

    def log_q(params_, pos_, spins_, atoms_, charges_):
      out = batch_network(params_, pos_, spins_, atoms_, charges_)
      kfac_jax.register_normal_predictive_distribution(out[:, None])
      return out.sum()

    score = jax.grad(log_q, argnums=1)
    primals = (params, data.positions, data.spins, data.atoms, data.charges)
    tangents = (
        tangents[0],
        tangents[2].positions,
        tangents[2].spins,
        tangents[2].atoms,
        tangents[2].charges,
    )
    score_primal, score_tangent = jax.jvp(score, primals, tangents)

    score_norm = jnp.linalg.norm(score_primal, axis=-1, keepdims=True)
    median = jnp.median(constants.all_gather(score_norm))
    deviation = jnp.mean(jnp.abs(score_norm - median))
    mask = score_norm < (median + 5 * deviation)
    log_q_tangent_out = (aux_data.grad_local_energy * score_tangent * mask).sum(
        axis=1
    )
    log_q_tangent_out *= len(mask) / mask.sum()

    if vmc_weight > 1e-9:
      _, psi_tangent = jax.jvp(batch_network, primals, tangents)
      log_q_tangent_out += vmc_weight * diff * psi_tangent
    primals_out = loss, aux_data
    tangents_out = (log_q_tangent_out.mean(), aux_data)
    return primals_out, tangents_out

  return total_energy


def make_energy_overlap_loss(network: networks.LogFermiNetLike,
                             local_energy: hamiltonian.LocalEnergy,
                             clip_local_energy: float = 0.0,
                             clip_from_median: bool = True,
                             center_at_clipped_energy: bool = True,
                             overlap_penalty: float = 1.0,
                             overlap_weight: Tuple[float, ...] = (1.0,),
                             complex_output: bool = False,
                             max_vmap_batch_size: int = 0) -> LossFn:
  """Creates the loss function for the penalty method for excited states.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at the
      single MCMC configuration given the network parameters. For the overlap
      loss, this returns an entire state matrix - all pairs of states and
      walkers.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate.
    center_at_clipped_energy: If true, center the local energy differences
      passed back to the gradient around the clipped local energy, so the mean
      difference across the batch is guaranteed to be zero. Seems to
      significantly improve performance with pseudopotentials.
    overlap_penalty: The strength of the penalty term that controls the
      tradeoff between minimizing the weighted energies and keeping the states
      orthogonal.
    overlap_weight: The weight to apply to each individual energy in the overall
      optimization.
    complex_output: If true, the network output is complex-valued.
    max_vmap_batch_size: If 0, use standard vmap. If >0, use batched_vmap with
      the given batch size.

  Returns:
    LossFn callable which evaluates the total energy of the system.
  """

  vmap = jax.vmap if max_vmap_batch_size == 0 else functools.partial(
      folx.batched_vmap, max_batch_size=max_vmap_batch_size)
  data_axes = networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0)
  batch_local_energy = vmap(
      local_energy, in_axes=(None, 0, data_axes), out_axes=(0, 0))
  batch_network = vmap(network, in_axes=(None, 0, 0, 0, 0), out_axes=0)
  overlap_weight = jnp.array(overlap_weight)

  # TODO(pfau): how much of this can be factored out with make_loss?
  @jax.custom_jvp
  def total_energy_and_overlap(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the energy of the network for a batch of configurations."""

    # Energy term. Largely similar to make_energy_loss, but simplified.
    keys = jax.random.split(key, num=data.positions.shape[0])
    e_l, _ = batch_local_energy(params, keys, data)
    loss = constants.pmean(jnp.mean(e_l, axis=0))
    loss_diff = e_l - loss
    variance = constants.pmean(
        jnp.mean(loss_diff * jnp.conj(loss_diff), axis=0))
    weighted_energy = jnp.dot(loss, overlap_weight)

    # Overlap matrix. To compute S_ij^2 = <psi_i psi_j>^2/<psi_i^2><psi_j^2>
    # by Monte Carlo, you can split up the terms into a product of MC estimates
    # E_{x_i ~ psi_i^2} [ psi_j(x_i) / psi_i(x_i) ] *
    # E_{x_j ~ psi_j^2} [ psi_i(x_j) / psi_j(x_j) ]
    # Since x_i and x_j are sampled independently, the product of empirical
    # estimates is an unbiased estimate of the product of expectations.

    # #TODO(pfau): Avoid the double call to batch_network here and in the jvp.
    sign_psi, log_psi = batch_network(params,
                                      data.positions,
                                      data.spins,
                                      data.atoms,
                                      data.charges)
    sign_psi_diag = jax.vmap(jnp.diag)(sign_psi)[..., None]
    log_psi_diag = jax.vmap(jnp.diag)(log_psi)[..., None]
    s_ij_local = sign_psi * sign_psi_diag * jnp.exp(log_psi - log_psi_diag)
    s_ij = constants.pmean(jnp.mean(s_ij_local, axis=0))
    total_overlap = jnp.sum(jnp.triu(s_ij * s_ij.T, 1))

    return (weighted_energy + overlap_penalty * total_overlap,
            AuxiliaryLossData(
                energy=loss,
                variance=variance.real,
                local_energy=e_l,
                clipped_energy=loss,
                s_ij=s_ij_local,
                mean_s_ij=s_ij,
                local_energy_mat=e_l))

  @total_energy_and_overlap.defjvp
  def total_energy_and_overlap_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    if complex_output:
      raise NotImplementedError('Complex output is not supported with penalty '
                                'method gradients for excited states.')

    params, key, data = primals
    batch_loss, aux_data = total_energy_and_overlap(params, key, data)
    energy = aux_data.energy.real

    if clip_local_energy > 0.0:
      clipped_energy, energy_diff = clip_local_values(
          aux_data.local_energy,
          energy,
          clip_local_energy,
          clip_from_median,
          center_at_clipped_energy)
      aux_data.clipped_energy = jnp.dot(clipped_energy, overlap_weight)
    else:
      energy_diff = aux_data.local_energy - energy

    # To take the gradient of the overlap squared between psi_i and psi_j, we
    # can use a similar derivation to the gradient of the energy, which gives
    # \nabla_i S_ij^2 =
    # 2 E_{x_j ~ psi_j^2} [ psi_i(x_j) / psi_j(x_j) ] *
    # E_{x_i ~ psi_i^2} [ (psi_j(x_i) / psi_i(x_i) - <psi_j(x_i) / psi_i(x_i)>)
    #                     \nabla_i log psi_i(x_i) ]
    # where \nabla_i means the gradient wrt the parameters of psi_i
    # Again, because the two expectations are taken over independent samples,
    # the product of empirical estimates will be unbiased.

    if clip_local_energy > 0.0:
      clipped_overlap, overlap_diff = clip_local_values(
          aux_data.s_ij,
          aux_data.mean_s_ij,
          clip_local_energy,
          clip_from_median,
          center_at_clipped_energy)
    else:
      clipped_overlap = aux_data.s_ij
      overlap_diff = clipped_overlap - aux_data.mean_s_ij

    overlap_diff = 2 * jnp.sum(jnp.triu(
        clipped_overlap * overlap_diff.transpose((0, 2, 1)), 1), axis=1)

    # Due to the simultaneous requirements of KFAC (calling convention must be
    # (params, rng, data)) and Laplacian calculation (only want to take
    # Laplacian wrt electron positions) we need to change up the calling
    # convention between total_energy and batch_network
    data = primals[2]
    data_tangents = tangents[2]
    primals = (primals[0], data.positions, data.spins, data.atoms, data.charges)
    tangents = (tangents[0], data_tangents.positions, data_tangents.spins,
                data_tangents.atoms, data_tangents.charges)

    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    _, log_primal = psi_primal
    _, log_tangent = psi_tangent
    kfac_jax.register_normal_predictive_distribution(
        jax.vmap(jnp.diag)(log_primal))
    device_batch_size = jnp.shape(aux_data.local_energy)[0]
    tangent_loss = energy_diff * overlap_weight + overlap_penalty * overlap_diff
    tangents_out = (
        jnp.sum(jax.vmap(jnp.diag)(log_tangent) * tangent_loss) /
        device_batch_size,
        aux_data)

    primals_out = batch_loss.real, aux_data
    return primals_out, tangents_out

  return total_energy_and_overlap
