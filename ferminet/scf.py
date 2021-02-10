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

"""Interaction with Hartree-Fock solver in pyscf."""

# Abbreviations used:
# SCF: self-consistent field (method). Another name for Hartree-Fock
# HF: Hartree-Fock method.
# RHF: restricted Hartre-Fock. Require molecular orbital for the i-th alpha-spin
#   and i-th beta-spin electrons to have the same spatial component.
# ROHF: restricted open-shell Hartree-Fock. Same as RHF except allows the number
#   of alpha and beta electrons to differ.
# UHF: unrestricted Hartre-Fock. Permits breaking of spin symmetry and hence
#   alpha and beta electrons to have different spatial components.
# AO: Atomic orbital. Underlying basis set (typically Gaussian-type orbitals and
#   built into pyscf).
# MO: molecular orbitals/Hartree-Fock orbitals. Single-particle orbitals which
#   are solutions to the Hartree-Fock equations.


from typing import Callable, Tuple

from ferminet.utils import scf as base_scf
import tensorflow.compat.v1 as tf


class Scf(base_scf.Scf):
  """Helper class for running Hartree-Fock (self-consistent field) with pyscf.

  This extends ferminet.utils.scf.Scf with tensorflow ops.

  Attributes:
    molecule: list of system.Atom objects giving the atoms in the molecule
      and their positions.
    nelectrons: Tuple with number of alpha electrons and beta electrons.
    basis: Basis set to use, best specified with the relevant string for a
      built-in basis set in pyscf. A user-defined basis set can be used
      (advanced). See https://sunqm.github.io/pyscf/gto.html#input-basis for
        more details.
    restricted: If true, use the restriced Hartree-Fock method, otherwise use
      the unrestricted Hartree-Fock method.
  """

  def tf_eval_mos(self, positions: tf.Tensor, deriv: bool = False,
                 ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
    """Evaluates the Hartree-Fock single-particle orbitals as a tensorflow op.

    See: `eval_mos` for evaluating the orbitals given positions as numpy arrays.

    Args:
      positions: Tensor of shape (N, 3) containing N 3D position vectors at
        which the Hartree-Fock orbitals are evaluated.
      deriv: If True, also evaluates the first derivatives of the orbitals with
        respect to positions and makes them available via tf.gradients.

    Returns:
      Tensor of shape (N, 2M) and the same dtype as positions. The first
      (second) M elements along the last axis gives the value of the alpha-spin
      (beta-spin) Hartree-Fock orbitals at the desired positions.

    Raises:
      RuntimeError: If the first derivatives are requested but deriv is False.
    """

    # Evaluate MOs in a nested function to avoid issues with custom_gradients
    # and self or with the deriv flag argument.

    @tf.custom_gradient
    def _eval_mos(positions: tf.Tensor
                 ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
      """Computes Hartree-Fock orbitals at given positions."""
      mo_values_derivs = tf.py_func(self.eval_mos, [positions, deriv],
                                    (tf.float64, tf.float64))
      mo_values_derivs = [
          tf.cast(vals, positions.dtype) for vals in mo_values_derivs
      ]
      if deriv:
        mo_values = tf.concat([vals[0] for vals in mo_values_derivs], axis=-1)
        mo_derivs = tf.concat([vals[1:] for vals in mo_values_derivs], axis=-1)
      else:
        mo_values = tf.concat(mo_values_derivs, axis=-1)
        mo_derivs = None
      def grad(dmo: tf.Tensor) -> tf.Tensor:
        """Computes gradients of orbitals with respect to positions."""
        # dmo, reverse sensitivity, is a (N, 2M) tensor.
        # mo_derivs is the full Jacobian, (3, N, 2M), where mo_derivs[i,j,k] is
        # d\phi_{k} / dr^j_i (derivative of the k-th orbital with respect to the
        # i-th component of the j-th position vector).
        # positions is a (N, 3) tensor => tf.gradients(mo, positions) is (N, 3).
        # tensorflow gradients use summation over free indices (i.e. orbitals).
        if mo_derivs is None:
          raise RuntimeError(
              'Gradients not computed in forward pass. Set derivs=True.')
        g = tf.reduce_sum(mo_derivs * tf.expand_dims(dmo, 0), axis=-1)
        return tf.linalg.transpose(g)
      return mo_values, grad

    return _eval_mos(positions)

  def tf_eval_hf(self, positions: tf.Tensor,
                 deriv: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
    """Evaluates Hartree-Fock occupied orbitals at a set of electron positions.

    Note: pyscf evaluates all orbitals at each position, which is somewhat
    wasteful for this use-case, where we only need occupied orbitals and further
    factorise the wavefunction into alpha and beta parts.

    Args:
      positions: (N, nelectrons, 3) tensor. Batch of positions for each
        electron. The first self.nelectrons[0] elements in the second axis refer
        to alpha electrons, the next self.electrons[1] positions refer to the
        beta electrons.
      deriv: If True, also evaluates the first derivatives of the orbitals with
        respect to positions and makes them available via tf.gradients.

    Returns:
      Tuple of tensors for alpha and beta electrons respectively of shape
      (N, nspin, nspin), where nspin is the corresponding number of electrons of
      that spin. The (i, j, k) element gives the value of the k-th orbital for
      the j-th electron in batch i. The product of the determinants of the
      tensors hence gives the Hartree-Fock determinant at the set of input
      positions.

    Raises:
      RuntimeError: If the sum of alpha and beta electrons does not equal
      nelectrons.
    """
    if sum(self.nelectrons) != positions.shape.as_list()[1]:
      raise RuntimeError(
          'positions does not contain correct number of electrons.')
    # Evaluate MOs at all electron positions.
    # pyscf evaluates all 2M MOs at each 3D position vector. Hance reshape into
    # (N*nelectron, 3) tensor for pyscf input.
    one_e_positions = tf.reshape(positions, (-1, positions.shape[-1]))
    mos = self.tf_eval_mos(one_e_positions, deriv)
    nbasis = tf.shape(mos)[-1] // 2
    # Undo reshaping of electron positions to give (N, nelectrons, 2M), where
    # the (i, j, k) element gives the value of the k-th MO at the j-th electron
    # position in the i-th batch.
    mos = tf.reshape(mos, (-1, sum(self.nelectrons), 2*nbasis))
    # Return (using Aufbau principle) the matrices for the occupied alpha and
    # beta orbitals. Number of alpha electrons given by electrons[0].
    alpha_spin = mos[:, :self.nelectrons[0], :self.nelectrons[0]]
    beta_spin = mos[:, self.nelectrons[0]:, nbasis:nbasis+self.nelectrons[1]]
    return alpha_spin, beta_spin

  def tf_eval_slog_slater_determinant(self, flat_positions: tf.Tensor
                                     ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Evaluates the signed log HF slater determinant at given flat_positions.

    Interface is chosen to be similar to that used for QMC networks.

    Args:
      flat_positions: (N, total_electrons * 3) tensor. Batch of flattened
        positions for each electron, reshapeable to (N, total_electrons, 3)
        where the first self.nelectrons[0] elements in the second axis refer to
        alpha electrons, the next self.nelectrons[1] positions refer to the beta
        electrons.

    Returns:
      log_abs_slater_determinant: (N, 1) tensor
      # log of absolute value of slater determinant
      sign: (N, 1) tensor. sign of wavefunction.

    Raises:
      RuntimeError: on incorrect shape of flat_positions.
    """
    xs = tf.reshape(flat_positions,
                    [flat_positions.shape[0],
                     sum(self.nelectrons), -1])
    if xs.shape[2] != 3:
      msg = 'flat_positions must be of shape (N, total_electrons*3)'
      raise RuntimeError(msg)
    matrices = self.tf_eval_hf(xs)
    slogdets = [tf.linalg.slogdet(elem) for elem in matrices]
    sign_alpha, sign_beta = [elem[0] for elem in slogdets]
    log_abs_wf_alpha, log_abs_wf_beta = [elem[1] for elem in slogdets]
    log_abs_slater_determinant = tf.expand_dims(tf.math.add(log_abs_wf_alpha,
                                                            log_abs_wf_beta), 1)
    sign = tf.expand_dims(tf.math.multiply(sign_alpha, sign_beta), 1)
    return log_abs_slater_determinant, sign

  def tf_eval_slog_hartree_product(self, flat_positions: tf.Tensor,
                                   deriv: bool = False
                                  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Evaluates the signed log Hartree product without anti-symmetrization.

    Interface is chosen to be similar to that used for QMC networks.

    Args:
      flat_positions: (N, total_electrons * 3) tensor. Batch of flattened
        positions for each electron, reshapeable to (N, total_electrons, 3)
        where the first self.nelectrons[0] elements in the second axis refer to
        alpha electrons, the next self.nelectrons[1] positions refer to the beta
        electrons.
      deriv: specifies if we will need derivatives.

    Returns:
      log_abs_hartree_product: (N, 1) tensor. log of abs of Hartree product.
      sign: (N, 1) tensor. sign of Hartree product.
    """
    xs = tf.reshape(flat_positions,
                    [int(flat_positions.shape[0]),
                     sum(self.nelectrons), -1])
    matrices = self.tf_eval_hf(xs, deriv)
    log_abs_alpha, log_abs_beta = [
        tf.linalg.trace(tf.log(tf.abs(elem))) for elem in matrices
    ]
    sign_alpha, sign_beta = [
        tf.reduce_prod(tf.sign(tf.linalg.diag_part(elem)), axis=1)
        for elem in matrices
    ]
    log_abs_hartree_product = tf.expand_dims(
        tf.math.add(log_abs_alpha, log_abs_beta), axis=1)
    sign = tf.expand_dims(tf.math.multiply(sign_alpha, sign_beta), axis=1)
    return log_abs_hartree_product, sign

  def tf_eval_hartree_product(self, flat_positions: tf.Tensor,
                              deriv: bool = False) -> tf.Tensor:
    """Evaluates the signed log Hartree product without anti-symmetrization.

    Interface is chosen to be similar to that used for QMC networks. This is
    a convenience wrapper around tf_eval_slog_hartree_product and is hence not
    optimised.

    Args:
      flat_positions: (N, total_electrons * 3) tensor. Batch of flattened
        positions for each electron, reshapeable to (N, total_electrons, 3)
        where the first self.nelectrons[0] elements in the second axis refer to
        alpha electrons, the next self.nelectrons[1] positions refer to the beta
        electrons.
      deriv: specifies if we will need derivatives.

    Returns:
      (N,1) tensor containing the Hartree product.
    """
    log_abs, sign = self.tf_eval_slog_hartree_product(flat_positions, deriv)
    return tf.exp(log_abs) * sign
