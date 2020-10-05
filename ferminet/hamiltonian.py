# Lint as: python3
# Copyright 2018 DeepMind Technologies Limited. All Rights Reserved.
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

"""Utilities for constructing Hamiltonians and setting up VMC calculations."""

import numpy as np
import tensorflow.compat.v1 as tf


def kinetic_from_log(f, x):
  r"""Compute -1/2 \nabla^2 \psi / \psi from log|psi|."""
  with tf.name_scope('kinetic_from_log'):
    df = tf.gradients(f, x)[0]
    lapl_elem = tf.map_fn(
        lambda i: tf.gradients(tf.expand_dims(df[..., i], -1), x)[0][..., i],
        tf.range(x.shape[1]),
        dtype=x.dtype.base_dtype,
        back_prop=False,
        parallel_iterations=20)
    lapl = (tf.reduce_sum(lapl_elem, axis=0) +
            tf.reduce_sum(df**2, axis=-1))
    return -0.5*tf.expand_dims(lapl, -1)


def operators(atoms,
              nelectrons,
              potential_epsilon=0.0):
  """Creates kinetic and potential operators of Hamiltonian in atomic units.

  Args:
    atoms: list of Atom objects for each atom in the system.
    nelectrons: number of electrons
    potential_epsilon: Epsilon used to smooth the divergence of the 1/r
      potential near the origin for algorithms with numerical stability issues.

  Returns:
    The functions that generates the kinetic and potential energy as a TF op.
  """
  vnn = 0.0
  for i, atom_i in enumerate(atoms):
    for atom_j in atoms[i+1:]:
      qij = float(atom_i.charge * atom_j.charge)
      vnn += qij / np.linalg.norm(atom_i.coords_array - atom_j.coords_array)

  def smooth_norm(x):
    with tf.name_scope('smooth_norm'):
      if potential_epsilon == 0.0:
        return tf.norm(x, axis=1, keepdims=True)
      else:
        return tf.sqrt(tf.reduce_sum(x**2 + potential_epsilon**2,
                                     axis=1,
                                     keepdims=True))

  def nuclear_potential(xs):
    """Calculates nuclear potential for set of electron positions."""
    with tf.name_scope('Vne'):
      v = []
      for atom in atoms:
        charge = tf.constant(atom.charge, dtype=xs[0].dtype)
        coords = tf.constant(atom.coords, dtype=xs[0].dtype)
        v.extend([-charge / smooth_norm(coords - x) for x in xs])
      return tf.add_n(v)

  def electronic_potential(xs):
    """Calculates electronic potential for set of electron positions."""
    with tf.name_scope('Vee'):
      if len(xs) > 1:
        v = []
        for (i, ri) in enumerate(xs):
          v.extend([
              1.0 / smooth_norm(ri - rj) for rj in xs[i + 1:]
          ])
        return tf.add_n(v)
      else:
        return 0.0

  def nuclear_nuclear(dtype):
    """Calculates the nuclear-nuclear interaction contribution."""
    with tf.name_scope('Vnn'):
      return tf.constant(vnn, dtype=dtype)

  def potential(x):
    """Calculates the total potential energy at each electron position."""
    xs = tf.split(x, nelectrons, axis=1)
    return (nuclear_potential(xs) + electronic_potential(xs)
            + nuclear_nuclear(xs[0].dtype))

  return kinetic_from_log, potential


def exact_hamiltonian(atoms,
                      nelectrons,
                      potential_epsilon=0.0):
  """Construct function that evaluates exact Hamiltonian of a system.

  Args:
    atoms: list of Atom objects for each atom in the system.
    nelectrons: number of electrons
    potential_epsilon: Epsilon used to smooth the divergence of the 1/r
      potential near the origin for algorithms with numerical stability issues.

  Returns:
    A functions that generates the wavefunction and hamiltonian op.
  """
  k_fn, v_fn = operators(atoms, nelectrons, potential_epsilon)
  def _hamiltonian(f, x):
    # kinetic_from_log actually computes -1/2 \nabla^2 f / f, so multiply by f
    logpsi, signpsi = f(x)
    psi = tf.exp(logpsi) * signpsi
    hpsi = psi * (k_fn(logpsi, x) + v_fn(x))
    return psi, hpsi
  return _hamiltonian


def r12_features(x, atoms, nelectrons, keep_pos=True, flatten=False,
                 atomic_coords=False):
  """Adds physically-motivated features depending upon electron distances.

  The tensor of electron positions is extended to include the distance of each
  electron to each nucleus and distance between each pair of electrons.

  Args:
    x: electron positions. Tensor either of shape (batch_size, nelectrons*ndim),
      or (batch_size, nelectrons, ndim), where ndim is the dimensionality of the
      system (usually 3).
    atoms: list of Atom objects for each atom in the system.
    nelectrons: number of electrons.
    keep_pos: If true, includes the original electron positions in the output
    flatten: If true, return the distances as a flat vector for each element of
      the batch. If false, return the atom-electron distances and electron-
      electron distances each as 3D arrays.
    atomic_coords: If true, replace the original position of the electrons with
      the position of the electrons relative to all atoms.

  Returns:
    If flatten is true, keep_pos is true and atomic_coords is false:
      tensor of shape (batch_size, ndim*Ne + Ne*Na + Ne(Ne-1)/2), where Ne (Na)
      is the number of electrons (atoms). The first ndim*Ne terms are the
      original x, the next Ne terms are  |x_i - R_1|, where R_1 is the position
      of the first nucleus (and so on for each atom), and the remaining terms
      are |x_i - x_j| for each (i,j) pair, where i and j run over all electrons,
      with i varied slowest.
    If flatten is true and keep_pos is false: it does not include the first
      ndim*Ne features.
    If flatten is false and keep_pos is false: tensors of shape
      (batch_size, Ne, Na) and (batch_size, Ne, Ne)
    If flatten is false and keep_pos is true: same as above, and also a tensor
      of size (batch_size, Ne, ndim)
    If atomic_coords is true: the same as if keep_pos is true, except the
      ndim*Ne coordinates corresponding to the original positions are replaced
      by ndim*Ne*Na coordinates corresponding to the different from each
      electron position to each atomic position.
  """
  with tf.name_scope('r12_features'):
    if len(x.shape) == 2:
      xs = tf.reshape(x, [x.shape[0], nelectrons, -1])
    else:
      xs = x
    coords = tf.stack([
        tf.constant(atom.coords, dtype=x.dtype.base_dtype) for atom in atoms
    ])
    coords = tf.expand_dims(tf.expand_dims(coords, 0), 0)
    xsa = tf.expand_dims(xs, 2) - coords  # xs in atomic coordinates

    r_ae = tf.norm(xsa, axis=-1)
    r_ee = np.zeros((nelectrons, nelectrons), dtype=object)
    for i in range(nelectrons):
      for j in range(i+1, nelectrons):
        r_ee[i, j] = tf.norm(xs[:, i, :] - xs[:, j, :], axis=1, keepdims=True)

    if flatten:
      r_ae = tf.reshape(r_ae, [r_ae.shape[0], -1])
      if nelectrons > 1:
        r_ee = tf.concat(
            r_ee[np.triu_indices(nelectrons, k=1)].tolist(), axis=1)
      else:
        r_ee = tf.zeros([r_ae.shape[0], 0])
      if keep_pos:
        if atomic_coords:
          xsa = tf.reshape(xsa, [xsa.shape[0], -1])
          return tf.concat([r_ae, r_ee, xsa], axis=1)
        else:
          return tf.concat([r_ae, r_ee, x], axis=1)
      else:
        return tf.concat([r_ae, r_ee], axis=1)
    else:
      zeros_like = tf.zeros((xs.shape[0], 1), dtype=x.dtype.base_dtype)
      for i in range(nelectrons):
        r_ee[i, i] = zeros_like
        for j in range(i):
          r_ee[i, j] = r_ee[j, i]
      r_ee = tf.transpose(tf.stack(r_ee.tolist()), [2, 0, 1, 3])
      if keep_pos:
        if atomic_coords:
          return r_ae, r_ee, xsa
        else:
          return r_ae, r_ee, xs
      else:
        return r_ae, r_ee
