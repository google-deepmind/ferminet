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
"""Network building for quantum Monte Carlo calculations."""

# Using networks with MCMC and optimisation
#
# We create different instances of the same model, all using the same parameters
# via Sonnet's variable sharing: one to run pre-training, one to sample electron
# configurations using MCMC and one for use with the optimiser. This requires
# care when using K-FAC: normally one would only have multiple registrations of
# network instances if using multi-tower mode, which we aren't.
# K-FAC detects variable re-use from the variable scope, which gets set to true
# by Sonnet when we create additional instances of the model, in layer
# registration, and hence results in errors because we're not using multi-tower.
# Thus we override the defaults by explicitly setting reuse=False in all layer
# registrion calls.

# pylint: disable=g-long-lambda

import functools
import typing

from ferminet import hamiltonian
import sonnet as snt
import tensorflow.compat.v1 as tf


def extend(x, y):
  """Add extra dimensions so x can be broadcast with y."""
  if isinstance(x, float):
    return x
  rank = y.shape.rank
  rank_ = x.shape.rank
  assert rank_ <= rank
  for _ in range(rank - rank_):
    x = tf.expand_dims(x, -1)
  return x


def flatten(seq):
  """Recursively flattens a sequence."""
  lst = []
  for el in seq:
    if isinstance(el, (list, tuple)):
      lst.extend(flatten(el))
    else:
      lst.append(el)
  return lst


@tf.custom_gradient
def sign_det(x):
  """Sign of determinant op whose gradient doesn't crash when the sign is 0."""
  with tf.name_scope('sign_det'):
    sign = tf.linalg.slogdet(x)[0]

    def grad(_):
      return tf.zeros_like(x)

  return sign, grad


@tf.custom_gradient
def determinant(x):
  """Determinant op with well-defined gradient for singular matrices."""
  with tf.name_scope('determinant'):
    with tf.device('/cpu:0'):
      s, u, v = tf.linalg.svd(x)

    sign = sign_det(x)

    def grad(dx):
      ss = tf.tile(s[..., None], len(s.shape) * [1] + [s.shape[-1]])
      ss = tf.linalg.set_diag(ss, tf.ones_like(s))
      z = tf.reduce_prod(ss, axis=-2)
      adj = tf.matmul(u, tf.matmul(tf.linalg.diag(z), v, transpose_b=True))
      signdx = tf.expand_dims(tf.expand_dims(sign * dx, -1), -1)
      return signdx * adj

    return sign * tf.reduce_prod(s, axis=-1), grad


@tf.custom_gradient
def log_determinant(x):
  """Log-determinant op with well-defined gradient for singular matrices."""
  with tf.name_scope('determinant'):
    with tf.device('/cpu:0'):
      s, u, v = tf.linalg.svd(x)

    sign = sign_det(x)

    def grad(dsign, dlog):
      del dsign
      # d ln|det(X)|/dX = transpose(X^{-1}). This uses pseudo-inverse via SVD
      # and so is not numerically stable...
      adj = tf.matmul(u,
                      tf.matmul(tf.linalg.diag(1.0 / s), v, transpose_b=True))
      return tf.expand_dims(tf.expand_dims(dlog, -1), -1) * adj

    return (sign, tf.reduce_sum(tf.log(s), axis=-1)), grad


def cofactor(u, s, v, shift=0.0):
  r"""Calculates the cofactor matrix from singular value decomposition.

  Given M = U S V*, where * indicates the conjugate transpose, the cofactor
  of M, M^C = det(A) A^{-T}, is given by M^C = det(U) det(V) U \Gamma V^T,
  where \Gamma_{ij} = \delta_{ij} \prod_{k!=i} s_k, s_k is the k-th singular
  value of M.  The latter equation for M^C is well behaved even for singular
  matrices.

  Note:

  det(U), det(V) \in {-1, +1} are calculated as part of the determinant
  evaluation and so not repeated here.

  Args:
    u: unitary matrix, U, from SVD.
    s: singular values, S, from SVD.
    v: unitary matrix, V, from SVD.
    shift (optional): constant to subtract in log domain for stability.

  Returns:
    cofactor matrix up to the sign given by det(U) det(V), i.e.
    M^C / (det(U) det(V)) = U \Gamma V^T, divided by exp(shift).
  """
  return tf.matmul(
      u, tf.matmul(tf.linalg.diag(gamma(s, shift)), v, transpose_b=True))


def gamma(s, shift=0.0):
  r"""Calculates diagonal of \Gamma_{ij} = \delta_{ij} \Prod_{k!=i} s_k.

  Args:
    s: singular values of matrix.
    shift (optional): constant to subtract in log domain for stability.

  Returns:
    Diagonal of \Gamma over the outer dimension of s, divided by exp(shift).
  """
  ls = tf.log(s)
  lower = tf.cumsum(ls, axis=-1, exclusive=True)
  upper = tf.cumsum(ls, axis=-1, exclusive=True, reverse=True)
  return tf.exp(lower + upper - shift)


def rho(s, shift=0.0):
  r"""Calculates \rho_ij = \prod_{k!=i,j} s_k, \rho_ii = 0.

  Args:
    s: singular values of matrix.
    shift (optional): constant to subtract in log domain for stability.

  Returns:
    \rho, the first derivative of \Gamma (see `gamma`), divided by exp(shift).
  """
  # Product of singular elements before and after two indices, i, j,
  # i.e. v_{ij} = \Prod_{k<i,l>j} s_i s_j -- outer product of forward and
  # reverse cumulative products. Valid only for j>i.
  # Computed in log domain for numerical stability.
  ls = tf.log(s)
  lower_cumsum = tf.cumsum(ls, axis=-1, exclusive=True)
  upper_cumsum = tf.cumsum(ls, axis=-1, reverse=True, exclusive=True)
  v = tf.expand_dims(lower_cumsum, -1) + tf.expand_dims(upper_cumsum, -2)

  # Product of singular elements between two indices, \Prod_{i<k<l} s_k.
  # Create matrix with strict upper diagonal set to s_k and ones elsewhere and
  # take cumulative product.
  # Also computed in log domain for numerical stability
  s_mat = tf.linalg.transpose(
      tf.tile(ls[..., None], [1] * len(s.shape) + [s.shape[-1]]))
  triu_s_mat = tf.linalg.set_diag(
      tf.matrix_band_part(s_mat, 0, -1), tf.zeros_like(s))
  z = tf.cumsum(triu_s_mat, exclusive=True, axis=-1)
  # Discard lower triangle.
  r = tf.exp(z + v - extend(shift, z))
  r = tf.matrix_band_part(r, 0, -1)
  r = tf.linalg.set_diag(r, tf.zeros_like(s))

  return r + tf.linalg.transpose(r)


def grad_cofactor(u, v, r, grad_dy):
  r"""Calculate the first derivative of the cofactor of a matrix M.

  Given M = U S V*, the cofactor of M, M^C, and \Gamma as defined in `cofactor`,
  d(M^C) = U (Tr(K \Gamma) S^{-1} - S^{-1} K \Gamma) V^T, where K = V^T dY^T U,
  and dY is the backwards sensitivities.  Whilst S^{-1} can be singular, such
  terms explicitly cancel, giving for the term in parantheses:

  ( ... ) = \sum_{k!=i} K_{kk} r_{ik}  for i = j
          = - K_{ik} r_{ik}            otherwise.

  where r is calculated by `rho`.

  Note:
    As in `cofactor`, the factor det(U) det(V) is not included.

  Args:
    u: unitary matrix U from SVD.
    v: unitary matrix V from SVD.
    r: matrix r_{ij} = \prod_{k!=i,j} s_k, where s_k is the k-th singular value.
      See `rho`.
    grad_dy: backwards sensitivites.

  Returns:
    first derivative of the cofactor matrix, up to the sign det(U) det(V).
  """
  with tf.name_scope('grad_cofactor'):
    k = tf.matmul(v, tf.matmul(grad_dy, u, transpose_a=True), transpose_a=True)
    # r_ii = 0 for convenience. Calculate off-diagonal terms.
    kr = -tf.multiply(k, r)
    # NB: r is symmetric and r_ii = 0.
    kr_diag = tf.matmul(r, tf.expand_dims(tf.linalg.diag_part(k), -1))
    kr = tf.linalg.set_diag(kr, tf.squeeze(kr_diag))
    return tf.matmul(u, tf.matmul(kr, v, transpose_b=True))


@tf.custom_gradient
def logdet_matmul(x1, x2, w):
  """Numerically stable implementation of log(abs(sum_i w_i |x1_i| * |x2_i|)).

  Args:
    x1: Tensor of shape [batch size, dets in, n, n].
    x2: Tensor of shape [batch size, dets in, m, m].
    w: Weight matrix of shape [dets in, dets out].

  Returns:
    Op that computes matmul(det(x1) * det(x2), w). For numerical stability,
    returns this op in the form of a tuple, the first of which is the log
    of the absolute value of the output, and the second of which is the sign of
    the output.

  Raises:
    ValueError if x1 and x2 do not have the same shape except the last two
    dimensions, or if the number of columns in det(x1) does not match the number
    of rows in w.
  """
  if x1.shape[:-2] != x2.shape[:-2]:
    raise ValueError('x1 and x2 must have the same number of determinants.')
  if w.shape[0] != x1.shape[-3]:
    raise ValueError('Number of determinants should equal '
                     'number of inputs to weights.')
  with tf.name_scope('logdet_matmul'):
    with tf.device('/cpu:0'):
      if x1.shape[-1] > 1:
        s1, u1, v1 = tf.linalg.svd(x1)
      if x2.shape[-1] > 1:
        s2, u2, v2 = tf.linalg.svd(x2)

    # Computing the sign of the determinant u and v instead of x means this
    # works even with singular matrices.
    if x1.shape[-1] > 1:
      sign1 = sign_det(u1) * sign_det(v1)
      logdet1 = tf.reduce_sum(tf.log(s1), axis=-1)
    else:
      sign1 = tf.sign(x1[..., 0, 0])
      logdet1 = tf.log(tf.abs(x1[..., 0, 0]))

    if x2.shape[-1] > 1:
      sign2 = sign_det(u2) * sign_det(v2)
      logdet2 = tf.reduce_sum(tf.log(s2), axis=-1)
    else:
      sign2 = tf.sign(x2[..., 0, 0])
      logdet2 = tf.log(tf.abs(x2[..., 0, 0]))

    sign = sign1 * sign2
    logdet = logdet1 + logdet2

    logdet_max1 = tf.reduce_max(logdet1, axis=-1, keepdims=True)
    logdet_max2 = tf.reduce_max(logdet2, axis=-1, keepdims=True)
    det = tf.exp(logdet - logdet_max1 - logdet_max2) * sign
    output = tf.matmul(det, w)
    sign_out = tf.sign(output)
    log_out = tf.log(tf.abs(output)) + logdet_max1 + logdet_max2

    @tf.custom_gradient
    def logdet_matmul_grad(x1, x2, w, grad_log):
      """Numerically stable gradient of log(abs(sum_i w_i |x1_i| * |x2_i|)).

      Args:
        x1: Tensor of shape [batch size, dets in, n, n].
        x2: Tensor of shape [batch size, dets in, m, m].
        w: Weight matrix of shape [dets in, dets out].
        grad_log: Tensor of shape [batch_size, dets_out] by which to
          left-multiply the Jacobian (aka the reverse sensitivity).

      Returns:
        Ops that compute the gradient of log(abs(matmul(det(x1) * det(x2), w))).
        The ops give the gradients with respect to the inputs x1, x2, w.
      """
      # This is missing a factor of exp(-logdet_max1-logdet_max2), which is
      # instead picked up by including it in the terms adj1*det2 and adj2*det1
      glog_out = grad_log / output
      dout = tf.matmul(glog_out, w, transpose_b=True)

      if x1.shape[-1] > 1:
        adj1 = cofactor(u1, s1, v1, extend(logdet_max1, s1)) * sign1[..., None,
                                                                     None]
      else:
        adj1 = tf.ones_like(x1) * extend(tf.exp(-logdet_max1), x1)

      if x2.shape[-1] > 1:
        adj2 = cofactor(u2, s2, v2, extend(logdet_max2, s2)) * sign2[..., None,
                                                                     None]
      else:
        adj2 = tf.ones_like(x2) * extend(tf.exp(-logdet_max2), x2)

      det1 = tf.exp(logdet1 - logdet_max1) * sign1
      det2 = tf.exp(logdet2 - logdet_max2) * sign2

      dx1 = adj1 * (det2 * dout)[..., None, None]
      dx2 = adj2 * (det1 * dout)[..., None, None]
      dw = tf.matmul(det, glog_out, transpose_a=True)

      def grad(grad_dx1, grad_dx2, grad_dw):
        """Stable gradient of gradient of log(abs(sum_i w_i |x1_i| * |x2_i|)).

        Args:
          grad_dx1: Tensor of shape [batch size, dets in, n, n].
          grad_dx2: Tensor of shape [batch size, dets in, m, m].
          grad_dw: Tensor of shape [dets in, dets out].

        Returns:
          Ops that compute the gradient of the gradient of
          log(abs(matmul(det(x1) * det(x2), w))). The ops give the gradients
          with respect to the outputs of the gradient op dx1, dx2, dw and the
          reverse sensitivity grad_log.
        """
        # Terms that appear repeatedly in different gradients.
        det_grad_dw = tf.matmul(det, grad_dw)
        # Missing a factor of exp(-2logdet_max1-2logdet_max2) which is included\
        # via terms involving r1, r2, det1, det2, adj1, adj2
        glog_out2 = grad_log / (output**2)

        adj_dx1 = tf.reduce_sum(adj1 * grad_dx1, axis=[-1, -2])
        adj_dx2 = tf.reduce_sum(adj2 * grad_dx2, axis=[-1, -2])

        adj_dx1_det2 = det2 * adj_dx1
        adj_dx2_det1 = det1 * adj_dx2
        adj_dx = adj_dx2_det1 + adj_dx1_det2

        if x1.shape[-1] > 1:
          r1 = rho(s1, logdet_max1)
          dadj1 = (grad_cofactor(u1, v1, r1, grad_dx1) * sign1[..., None, None])
        else:
          dadj1 = tf.zeros_like(x1)

        if x2.shape[-1] > 1:
          r2 = rho(s2, logdet_max2)
          dadj2 = (grad_cofactor(u2, v2, r2, grad_dx2) * sign2[..., None, None])
        else:
          dadj2 = tf.zeros_like(x2)

        # Computes gradients wrt x1 and x2.
        ddout_w = (
            tf.matmul(glog_out, grad_dw, transpose_b=True) -
            tf.matmul(glog_out2 * det_grad_dw, w, transpose_b=True))
        ddout_x = tf.matmul(
            glog_out2 * tf.matmul(adj_dx, w), w, transpose_b=True)

        grad_x1 = (
            adj1 * (det2 *
                    (ddout_w - ddout_x) + adj_dx2 * dout)[..., None, None] +
            dadj1 * (dout * det2)[..., None, None])
        grad_x2 = (
            adj2 * (det1 *
                    (ddout_w - ddout_x) + adj_dx1 * dout)[..., None, None] +
            dadj2 * (dout * det1)[..., None, None])

        adj_dx_w = tf.matmul(adj_dx, w)

        # Computes gradient wrt w.
        grad_w = (
            tf.matmul(adj_dx, glog_out, transpose_a=True) - tf.matmul(
                det, glog_out2 * (det_grad_dw + adj_dx_w), transpose_a=True))

        # Computes gradient wrt grad_log.
        grad_grad_log = (det_grad_dw + adj_dx_w) / output
        return grad_x1, grad_x2, grad_w, grad_grad_log

      return (dx1, dx2, dw), grad

    def grad(grad_log, grad_sign):
      """Computes gradient wrt log(w*det(x1)*det(x2))."""
      del grad_sign
      return logdet_matmul_grad(x1, x2, w, grad_log)

    return (log_out, sign_out), grad


class TensorMlp(snt.AbstractModule):
  """Regular MLP, but works with rank-2, -3, or -4 tensors as input."""

  def __init__(self,
               layers,
               rank=0,
               use_bias=True,
               activate_final=True,
               residual=False,
               gain=1.0,
               stddev=1.0,
               name='tensor_mlp'):
    """Create an MLP with different input ranks.

    Args:
      layers: list of ints, with number of units in each layer
      rank: Rank of the input, not counting batch or feature dimension. If 0,
        creates a normal MLP. If 1, uses 1D convolutions with kernel size 1. If
        2, uses 2D convolutions with 1x1 kernel.
      use_bias: If false, leave bias out of linear layers.
      activate_final: Boolean. If true, add nonlinearity to last layer.
      residual: use residual connections in each layer.
      gain: gain used in orthogonal initializer for weights in each linear
        layer.
      stddev: standard deviation used in random normal initializer for biases in
        each linear layer.
      name: String. Sonnet default.

    Raises:
      ValueError: if the rank is not 0, 1 or 2.
    """
    super(TensorMlp, self).__init__(name=name)
    initializers = {'w': tf.initializers.orthogonal(gain=gain)}
    if use_bias:
      initializers['b'] = tf.initializers.random_normal(stddev=stddev)
    if rank == 0:
      self._linear = lambda x: snt.Linear(
          x, use_bias=use_bias, initializers=initializers)
    elif rank == 1:
      self._linear = lambda x: snt.Conv1D(
          output_channels=x,
          kernel_shape=1,
          use_bias=use_bias,
          initializers=initializers)
    elif rank == 2:
      self._linear = lambda x: snt.Conv2D(
          output_channels=x,
          kernel_shape=1,
          use_bias=use_bias,
          initializers=initializers)
    else:
      raise ValueError('Rank for TensorMlp must be 0, 1 or 2.')
    self._rank = rank
    self._nlayers = layers
    self._use_bias = use_bias
    self._activate_final = activate_final
    self._use_residual = residual

  def _build(self, inputs, layer_collection=None):
    if self._nlayers:
      x = inputs
      self._layers = []
      for i, h in enumerate(self._nlayers):
        self._layers.append(self._linear(h))
        y = self._layers[-1](x)
        if layer_collection:
          if self._use_bias:
            params = (self._layers[-1].w, self._layers[-1].b)
          else:
            params = self._layers[-1].w

          if self._rank == 0:
            layer_collection.register_fully_connected(params, x, y, reuse=False)
          elif self._rank == 1:
            layer_collection.register_conv1d(
                params, [1, 1, 1], 'SAME', x, y, reuse=False)
          else:
            layer_collection.register_conv2d(
                params, [1, 1, 1, 1], 'SAME', x, y, reuse=False)

        if i < len(self._nlayers) - 1 or self._activate_final:
          y = tf.nn.tanh(y)
        if self._use_residual and x.shape[-1].value == h:
          residual = x
        else:
          residual = 0.0
        x = y + residual
      return x
    return inputs  # if layers is an empty list, just act as the identity


class ParallelNet(snt.AbstractModule):
  """A symmetric network with parallel streams for rank 1 and 2 features.

  Rank 2 features propagate with parallel independent channels.
  Rank 1 features are able to input from other channels and rank 2 stream.

  Attributes:
    shape: number of alpha and beta electrons.
    layers: iterable containing integers pairs containing description of layers-
      i.e tuples containing (num_rank_one_features, num_rank_two_features)
    use_last_layer: bool. whether or not return rank-2 features at end of net.
    residual: bool. whether or not to use residual connections.
    name: for Sonnet namespacing.
  """

  def __init__(self,
               shape,
               layers,
               use_last_layer=True,
               residual=False,
               name='parallel_net'):
    super(ParallelNet, self).__init__(name=name)
    self._shape = shape  # tuple of input shapes
    self._n = sum(shape)
    self._layers = layers
    self._use_last_layer = use_last_layer
    self._use_residual = residual

  def _build(self, x1, x2, r_ee, layer_collection=None):
    """Construct the parallel network from rank 1 and rank 2 inputs.

    Args:
      x1: rank 1 input features. tensor of shape (batch, n_electrons,
        n_rank_one_inputs)
      x2: rank 2 input features. tensor of shape (batch, n_electrons,
        n_electrons, n_rank_two_inputs)
      r_ee: distances between electrons. tensor of shape (batch, n_electrons,
        n_electrons)
      layer_collection: KFAC layer collection or None.

    Returns:
      x1: rank 1 output features.
        tensor of shape (batch, n_electrons, n_rank_one_outputs)
      x2: rank 2 output features or None if use_last_layer is False.
        if not None tensor is of shape (batch, n_electrons, n_rank_two_outputs)
    """
    self._w1 = []
    self._w2 = []

    initializers = {
        'w': tf.initializers.orthogonal(),
        'b': tf.initializers.random_normal()
    }
    for i, (nc1, nc2) in enumerate(self._layers):
      xs1 = tf.split(x1, self._shape, axis=1)
      xs2 = tf.split(x2, self._shape, axis=1)

      # mean field features
      mf1 = [
          tf.tile(
              tf.reduce_mean(x, axis=1, keepdims=True), [1, x1.shape[1], 1])
          for x in xs1
      ]
      mf2 = [tf.reduce_mean(x, axis=1) for x in xs2]
      y1 = tf.concat([x1] + mf1 + mf2, axis=2)

      self._w1.append(
          snt.Conv1D(
              output_channels=nc1, kernel_shape=1, initializers=initializers))
      out1 = self._w1[-1](y1)

      if layer_collection:
        layer_collection.register_conv1d((self._w1[-1].w, self._w1[-1].b),
                                         [1, 1, 1],
                                         'SAME',
                                         y1,
                                         out1,
                                         reuse=False)

      # Implements residual connection
      if self._use_residual and x1.shape[-1].value == nc1:
        res = x1
      else:
        res = 0.0

      x1 = tf.nn.tanh(out1) + res

      # If we don't need the last layer of rank-2 features, set nc2 to None
      if i < len(self._layers) - 1 or self._use_last_layer:
        self._w2.append(
            snt.Conv2D(
                output_channels=nc2, kernel_shape=1, initializers=initializers))
        out2 = self._w2[-1](x2)

        if layer_collection:
          layer_collection.register_conv2d((self._w2[-1].w, self._w2[-1].b),
                                           [1, 1, 1, 1],
                                           'SAME',
                                           x2,
                                           out2,
                                           reuse=False)

        x2 = tf.nn.tanh(out2)
      else:
        x2 = None

    return x1, x2


class EquivariantNet(snt.AbstractModule):
  """A symmetric network like ParallelNet, but without the rank 2 stream."""

  def __init__(self, shape, layers, residual=False, name='parallel_net'):
    super(EquivariantNet, self).__init__(name=name)
    self._shape = shape  # tuple of input shapes
    self._n = sum(shape)
    self._layers = layers
    self._use_residual = residual

  def _build(self, x1, layer_collection=None):
    self._w1 = []
    initializers = {
        'w': tf.initializers.orthogonal(),
        'b': tf.initializers.random_normal()
    }
    for nc1 in self._layers:
      xs1 = tf.split(x1, self._shape, axis=1)

      # mean field features
      mf1 = [
          tf.tile(
              tf.reduce_mean(x, axis=1, keepdims=True), [1, x1.shape[1], 1])
          for x in xs1
      ]
      y1 = tf.concat([x1] + mf1, axis=2)

      self._w1.append(
          snt.Conv1D(
              output_channels=nc1, kernel_shape=1, initializers=initializers))
      out1 = self._w1[-1](y1)

      if layer_collection:
        layer_collection.register_conv1d((self._w1[-1].w, self._w1[-1].b),
                                         [1, 1, 1],
                                         'SAME',
                                         y1,
                                         out1,
                                         reuse=False)

      # Implements residual connection
      if self._use_residual and x1.shape[-1].value == nc1:
        res = x1
      else:
        res = 0.0

      x1 = tf.nn.tanh(out1) + res

    return x1


class LogMatmul(snt.AbstractModule):
  """Takes log-domain vector and performs numerically stable matmul."""

  def __init__(self, outputs, use_bias=False, name='log_matmul'):
    super(LogMatmul, self).__init__(name=name)
    self._outputs = outputs
    self._use_bias = use_bias

  def _build(self, log_input, sign_input, layer_collection=None):
    rank = log_input.shape.ndims
    if rank == 2:
      self._layer = snt.Linear(self._outputs, use_bias=self._use_bias)
    elif rank == 3:
      self._layer = snt.Conv1D(
          output_channels=self._outputs,
          kernel_shape=1,
          use_bias=self._use_bias)
    else:
      raise RuntimeError('LogMatmul only supports rank 2 and 3 tensors.')

    log_max = tf.reduce_max(log_input, axis=-1, keepdims=True)
    inputs = tf.exp(log_input - log_max) * sign_input
    output = self._layer(inputs)
    sign_output = tf.sign(output)
    log_output = tf.log(tf.abs(output)) + log_max
    if layer_collection:
      if self._use_bias:
        params = (self._layer.w, self._layer.b)
      else:
        params = self._layer.w
      if rank == 2:
        layer_collection.register_fully_connected(
            params, inputs, output, reuse=False)
      elif rank == 3:
        layer_collection.register_conv1d(
            params, [1, 1, 1], 'SAME', inputs, output, reuse=False)
    return log_output, sign_output


class Envelope(snt.AbstractModule):
  """Compute exponentially-decaying envelope to enforce boundary conditions."""

  def __init__(self, atoms, shape, determinants, soft=False, name='envelope'):
    r"""Compute exponentially-decaying envelope to enforce boundary conditions.

    Given a set of electron positions r_i and atom positions z_j, this module
    learns a set of Mahalonobis distance metrics A_jk and mixing proportions
    \pi_jk and computes a set of exponentially-decaying envelopes of the form:

    f_k(r_i) =
    \sum_j \pi_jk exp(-sqrt((r_i - z_j)^T * A_jk^T * A_jk * (r_i - z_j)))

    where the index `k` is over all orbitals, both spin up and spin down.
    Each of these envelopes is multiplied by one of the orbitals in a Fermi Net
    to enforce the boundary condition that the wavefunction goes to zero at
    infinity.

    Args:
      atoms: list of atom positions
      shape: tuple with number of spin up and spin down electrons
      determinants: number of determinants, used to compute number of envelopes
      soft: If true, use sqrt(1 + sum(x**2)) instead of norm(x)
      name: Sonnet boilerplate

    Returns:
      list of envelope ops, one spin up and one for spin down orbitals
    """
    super(Envelope, self).__init__(name=name)
    self._atoms = atoms
    self._shape = shape
    self._na = len(atoms)
    self._nd = determinants
    self._soft = soft  # remove cusp from envelope

  def _build(self, inputs, layer_collection=None):
    """Create ops to compute envelope for every electron/orbital pair.

    Args:
      inputs: tensor of shape (batch size, nelectrons, 3) with position of every
        electron
      layer_collection (optional): if using KFAC, the layer collection to
        register ops with.

    Returns:
      List of the value of the envelope for spin up and spin down electrons.
      Each element of the list is a tensorflow op with size
      (batch, shape[i], norbitals) where shape[i] is the number of electrons of
      spin "i", and norbitals is shape[i] * the number of determinants.
    """
    # shape of each (atom/orbital) envelope
    self._envelope_sigma = []
    # mixing proportion for all atoms for a given orbital
    self._envelope_pi = []
    self._envelopes = []
    norms = [[] for _ in self._shape]

    pos = tf.stack([
        tf.constant(atom.coords, dtype=inputs.dtype.base_dtype)
        for atom in self._atoms
    ])
    pos = tf.expand_dims(tf.expand_dims(pos, 0), 0)
    diff_ae = tf.expand_dims(inputs, 2) - pos

    # Split difference from electron to atom into list, one per atom, then
    # further split each element into a list, one for spin up electrons, one
    # for spin down.
    diff_aes = [
        tf.split(x, self._shape, axis=1)
        for x in tf.split(diff_ae, self._na, axis=2)
    ]

    # Compute the weighted distances from electron to atom
    # We swap the normal convention of `i` for the outer for loop and `j` for
    # the inner for loop to conform to the notation in the docstring.
    for j in range(self._na):  # iterate over atoms
      self._envelope_sigma.append([])
      for i in range(len(self._shape)):  # iterate over spin up/down electrons
        norbital = self._shape[i] * self._nd
        eyes = tf.transpose(
            tf.reshape(
                tf.eye(
                    3, batch_shape=[norbital], dtype=inputs.dtype.base_dtype),
                [1, 1, norbital * 3, 3]), (0, 1, 3, 2))
        self._envelope_sigma[j].append(
            tf.get_variable(
                'envelope_sigma_%d_%d' % (j, i),
                initializer=eyes,
                use_resource=True,
                trainable=True,
                dtype=inputs.dtype.base_dtype,
            ))

        # multiply the difference from the electron to the atom by an
        # anisotropic weight - one 3x3 matrix per orbital function

        # From the notation in the docstring, `rout` is
        # equivalent to (A_j1; ...; A_jk) * (r_i - z_j)

        # Note, snt.Conv2D won't take bare TF ops as initializers, so we just
        # use tf.nn.conv2d here instaed.
        rout = tf.nn.conv2d(diff_aes[j][i], self._envelope_sigma[j][i],
                            [1, 1, 1, 1], 'SAME')
        if layer_collection:
          layer_collection.register_conv2d(
              self._envelope_sigma[j][i], [1, 1, 1, 1],
              'SAME',
              diff_aes[j][i],
              rout,
              reuse=False)

        # Reshape `rout` so that rout[:, i, k, :] represents
        # A_jk * (r_i - z_j) for all r_i in the batch
        rout = tf.reshape(rout, [rout.shape[0], rout.shape[1], norbital, 3])
        if self._soft:
          norms[i].append(
              tf.sqrt(tf.reduce_sum(1 + rout**2, axis=3, keepdims=True)))
        else:
          norms[i].append(tf.norm(rout, axis=3, keepdims=True))

    # Compute the mixing proportions for different atoms
    for i in range(len(self._shape)):
      norbital = self._shape[i] * self._nd
      self._envelope_pi.append(
          tf.get_variable(
              'envelope_pi_%d' % i,
              initializer=tf.ones([norbital, self._na],
                                  dtype=inputs.dtype.base_dtype),
              use_resource=True,
              trainable=True,
              dtype=inputs.dtype.base_dtype,
          ))
      if layer_collection:
        layer_collection.register_generic(
            self._envelope_pi[i], inputs.shape[0].value, reuse=False)

      # Take the exponential of the Mahalonobis distance and multiply each
      # term by the weight for that atom.
      norm = tf.concat(norms[i], axis=3)
      mixed_norms = tf.exp(-norm) * self._envelope_pi[i]

      # Add the weighted terms for each atom together to form the envelope.
      # Each term in this list has shape (batch size, shape[i], norbitals)
      # where shape[i] is the number of electrons of spin `i`. This computes
      # the value of the envelope for every combination of electrons and
      # orbitals.
      self._envelopes.append(tf.reduce_sum(mixed_norms, axis=3))

    return self._envelopes


class BackFlow(snt.AbstractModule):
  """Introduce correlation in Slater network via coordinate transformations.

  Electron-electron, electron-nuclear and electron-electron-nuclear
  contributions are computed using MLPs for the nu, mu, theta and phi
  functions.
  """

  def _build(self, xs, xsa, r_en, r_ee, layer_collection=None):
    # Notation is explained in the CASINO manual, section 23.1
    # (https://casinoqmc.net/casino_manual_dir/casino_manual.pdf)
    backflow_ee = TensorMlp([64, 64, 64, 1],
                            rank=2,
                            activate_final=False,
                            residual=True,
                            gain=1.e-3,
                            stddev=1.e-3)

    backflow_en = TensorMlp([64, 64, 64, 1],
                            rank=2,
                            activate_final=False,
                            residual=True,
                            gain=1.e-3,
                            stddev=1.e-3)

    backflow_een = TensorMlp([64, 64, 64, 2],
                             rank=2,
                             activate_final=False,
                             residual=True,
                             gain=1.e-3,
                             stddev=1.e-3)

    # electron-electron backflow term
    eta = backflow_ee(r_ee, layer_collection=layer_collection)

    # electron-nuclear backflow term
    r_en = tf.expand_dims(r_en, -1)
    mu = backflow_en(r_en, layer_collection=layer_collection)

    # The most complicated term: the electron-electron-nuclear backflow term
    shape = [r_ee.shape[0], r_ee.shape[1], r_ee.shape[2], r_en.shape[2], 1]
    r_een = tf.concat((tf.broadcast_to(tf.expand_dims(
        r_ee, -2), shape), tf.broadcast_to(tf.expand_dims(r_en, -3), shape),
                       tf.broadcast_to(tf.expand_dims(r_en, -4), shape)),
                      axis=-1)
    # Reshape so we can squeeze this into a rank-2 network.
    r_een_flat = tf.reshape(r_een, [
        r_een.shape[0], r_een.shape[1], r_een.shape[2] * r_een.shape[3],
        r_een.shape[4]
    ])
    phi_theta = backflow_een(r_een_flat, layer_collection=layer_collection)
    phi = tf.reshape(phi_theta[..., 0], r_een.shape[:-1] + [1])
    theta = tf.reshape(phi_theta[..., 1], r_een.shape[:-1] + [1])

    diffs = tf.expand_dims(xs, 1) - tf.expand_dims(xs, 2)
    backflow = (
        tf.reduce_sum(eta * diffs, axis=2) + tf.reduce_sum(mu * xsa, axis=2) +
        tf.reduce_sum(phi * tf.expand_dims(diffs, axis=3), axis=[2, 3]) +
        tf.reduce_sum(theta * tf.expand_dims(xsa, axis=2), axis=[2, 3]))
    return backflow


class FermiNet(snt.AbstractModule):
  """Neural network with a determinant layer to antisymmetrize output."""

  def __init__(self,
               *,
               atoms,
               nelectrons,
               slater_dets,
               hidden_units,
               after_det,
               dim=3,
               architecture='ferminet',
               envelope=False,
               residual=False,
               r12_ee_features=True,
               r12_en_features=True,
               pos_ee_features=True,
               build_backflow=False,
               use_backflow=False,
               jastrow_en=False,
               jastrow_ee=False,
               jastrow_een=False,
               logdet=False,
               pretrain_iterations=2,
               name='det_net'):
    super(FermiNet, self).__init__(name=name)
    self._atoms = atoms
    self._dim = dim
    # tuple of number of (spin up, spin down) e's or a tuple of (e's) for
    # spin-polarised systems.
    self._shape = [nelec for nelec in nelectrons  if nelec > 0]
    self._ne = sum(nelectrons)  # total number of electrons
    self._na = len(atoms)  # number of atoms
    self._nd = slater_dets  # number of determinants

    self._architecture = architecture
    self._hidden_units = hidden_units
    self._after_det = after_det
    self._add_envelope = envelope
    self._r12_ee = r12_ee_features
    self._r12_en = r12_en_features
    self._pos_ee = pos_ee_features
    self._use_residual = residual
    self._build_backflow = build_backflow
    self._use_backflow = use_backflow
    self._use_jastrow_en = jastrow_en
    self._use_jastrow_ee = jastrow_ee
    self._use_jastrow_een = jastrow_een
    self._output_log = logdet  # Whether to represent wavef'n or log-wavef'n
    self.pretrain_iterations = pretrain_iterations

  def _features(self, xs, r_ee, r_en, xsa):
    x1 = []
    x2 = []
    if self._r12_en:
      x1.append(r_en)
    if self._r12_ee:
      x2.append(r_ee)
    xsa = tf.reshape(xsa, [xsa.shape[0], xsa.shape[1], -1])
    x1.append(xsa)
    if self._pos_ee:
      diffs = tf.expand_dims(xs, 1) - tf.expand_dims(xs, 2)
      x2.append(diffs)
    x1 = tf.concat(x1, axis=2) if x1 else None
    x2 = tf.concat(x2, axis=3) if x2 else None
    return x1, x2

  def _build(self, inputs, layer_collection=None, back_prop=True):
    self.inputs = inputs
    xs = tf.reshape(inputs, [inputs.shape[0], self._ne, -1])
    r_en, r_ee, xsa = hamiltonian.r12_features(
        inputs,
        self._atoms,
        self._ne,
        keep_pos=True,
        flatten=False,
        atomic_coords=True)

    if self._use_backflow or self._build_backflow:
      if self._architecture != 'slater':
        raise ValueError('Backflow should only be used with Hartree-Fock/'
                         'Slater-Jastrow network')
      backflow_net = BackFlow()
      backflow = backflow_net(xs, xsa, r_en, r_ee, layer_collection)
    else:
      backflow = None

    if backflow is not None:
      # Warning: Backflow transformed coordinates should *only* be fed into the
      # orbitals and not into the Jastrow factor. Denote functions of the
      # backflow as q* accordingly.
      q_inputs = inputs + tf.reshape(backflow, inputs.shape)
      qs = tf.reshape(q_inputs, [q_inputs.shape[0], self._ne, -1])
      q_en, q_ee, qsa = hamiltonian.r12_features(
          q_inputs,
          self._atoms,
          self._ne,
          keep_pos=True,
          flatten=False,
          atomic_coords=True)
      x1, x2 = self._features(qs, q_ee, q_en, qsa)
    else:
      x1, x2 = self._features(xs, r_ee, r_en, xsa)

    # Fermi Net or Slater-Jastrow net?
    if self._architecture == 'ferminet':
      if isinstance(self._hidden_units[0], typing.Iterable):
        self._parallel_net = ParallelNet(
            self._shape,
            self._hidden_units,
            residual=self._use_residual,
            use_last_layer=False)
        self._before_det, self._before_jastrow = self._parallel_net(
            x1, x2, r_ee, layer_collection=layer_collection)
      else:
        self._parallel_net = EquivariantNet(
            self._shape, self._hidden_units, residual=self._use_residual)
        self._before_det = self._parallel_net(
            x1, layer_collection=layer_collection)
        self._before_jastrow = None
    elif self._architecture == 'slater':
      self._before_det_mlp = TensorMlp(
          self._hidden_units, rank=1, residual=True)
      self._before_det = self._before_det_mlp(
          x1, layer_collection=layer_collection)
    else:
      raise ValueError('Not a recognized architecture: {}'.format(
          self._architecture))

    # Split the features into spin up and down electrons, and shape into
    # orbitals and determinants.
    before_det = tf.split(self._before_det, self._shape, axis=1)
    self._orbital_layers = [
        snt.Conv1D(x.shape[1] * self._nd, 1, name='orbital_%d' % i)
        for i, x in enumerate(before_det)
    ]
    orbitals = []
    for i, x in enumerate(before_det):
      layer = self._orbital_layers[i]
      y = layer(x)
      orbitals.append(y)
      if layer_collection:
        layer_collection.register_conv1d((layer.w, layer.b), [1, 1, 1],
                                         'SAME',
                                         x,
                                         y,
                                         reuse=False)

    # Create an exponentially-decaying envelope around every atom
    if self._add_envelope:
      envelope = Envelope(self._atoms, self._shape, self._nd)
      self._envelopes = envelope(xs, layer_collection=layer_collection)
      for i in range(len(orbitals)):
        orbitals[i] = orbitals[i] * self._envelopes[i]

    self._orbitals = [
        tf.transpose(
            tf.reshape(x, [x.shape[0], x.shape[1], -1, x.shape[1]]),
            [0, 2, 1, 3]) for x in orbitals
    ]

    slog1d = lambda x: (tf.sign(x[..., 0, 0]), tf.log(tf.abs(x[..., 0, 0])))
    slog = lambda x: log_determinant(x) if x.shape[-1].value > 1 else slog1d(x)
    if self._output_log:
      if self._nd == 1:
        self._slogdets = [
            slog(x) for x in self._orbitals
        ]
        self._logdet = tf.add_n([x[1] for x in self._slogdets])
        self._signs = functools.reduce(
            tf.Tensor.__mul__,  # pytype: disable=unsupported-operands
            [x[0] for x in self._slogdets])
        output = (self._logdet, self._signs)
      else:
        self._after_det_weights = tf.get_variable(
            'after_det_weights',
            shape=[self._nd, self._after_det[0]],
            initializer=tf.initializers.ones(dtype=inputs.dtype.base_dtype),
            use_resource=True,
            dtype=inputs.dtype.base_dtype,
        )
        if back_prop:
          if len(self._orbitals) == 1:
            output = list(slog(x) for x in self._orbitals)
            signdet, logdet = output[0]
            log_max = tf.reduce_max(logdet, axis=-1, keepdims=True)
            inputs = tf.exp(logdet - log_max) * signdet
            output = tf.matmul(inputs, self._after_det_weights)
            output = tf.log(tf.abs(output)) + log_max, tf.sign(output)
          else:
            output = logdet_matmul(self._orbitals[0], self._orbitals[1],
                                   self._after_det_weights)
        else:
          # Compute logdet with tf.linalg.slogdet, as gradients are not needed
          # We keep this outside logdet_matmul, despite the code duplication,
          # because tf.custom_gradient decorator does not like keyword args
          if len(self._orbitals) == 1:
            signdet, logdet = tf.linalg.slogdet(self._orbitals[0])
          else:
            slogdet0 = tf.linalg.slogdet(self._orbitals[0])
            slogdet1 = tf.linalg.slogdet(self._orbitals[1])
            signdet = slogdet0[0] * slogdet1[0]
            logdet = slogdet0[1] + slogdet1[1]

          log_max = tf.reduce_max(logdet, axis=-1, keepdims=True)
          inputs = tf.exp(logdet - log_max) * signdet
          output = tf.matmul(inputs, self._after_det_weights)
          output = tf.log(tf.abs(output)) + log_max, tf.sign(output)

        self._logdet = output[0]
        if layer_collection:
          layer_collection.register_generic(
              self._after_det_weights,
              self._orbitals[0].shape[0].value,
              reuse=False)

        if len(self._after_det) > 1:
          log_out, sign_out = output
          # Constructing x and -x to feed through network to make odd function
          output = (tf.stack((log_out, log_out),
                             axis=1), tf.stack((sign_out, -1 * sign_out),
                                               axis=1))
          for nout in self._after_det[1:]:
            layer = LogMatmul(nout, use_bias=True)
            # apply ReLU in log domain
            output = (tf.where(
                tf.greater(output[1], 0.0), output[0],
                -100 * tf.ones_like(output[0])), output[1])
            output = layer(*output, layer_collection=layer_collection)

          # Make output an odd function
          log_out, sign_out = output
          logs_out = [
              tf.squeeze(out, axis=1) for out in tf.split(log_out, 2, axis=1)
          ]
          signs_out = [
              tf.squeeze(out, axis=1) for out in tf.split(sign_out, 2, axis=1)
          ]
          log_max = tf.maximum(
              tf.reduce_max(logs_out[0], axis=1, keepdims=True),
              tf.reduce_max(logs_out[1], axis=1, keepdims=True))
          real_out = (
              tf.exp(logs_out[0] - log_max) * signs_out[0] -
              tf.exp(logs_out[1] - log_max) * signs_out[1])
          output = (tf.log(tf.abs(real_out)) + log_max, tf.sign(real_out))
    else:
      self._dets = [
          determinant(x) if x.shape[-1].value > 1 else x[..., 0, 0]
          for x in self._orbitals
      ]
      self._det = functools.reduce(tf.Tensor.__mul__, self._dets)  # pytype: disable=unsupported-operands
      after_det_input = self._det

      self._after_det_mlp = TensorMlp(self._after_det, use_bias=True)

      combined_input = tf.concat([after_det_input, -after_det_input], axis=0)
      combined_output = self._after_det_mlp(
          combined_input, layer_collection=layer_collection)
      output_0, output_1 = tf.split(combined_output, 2, axis=0)
      output = output_0 - output_1

    total_jastrow = 0.0
    if self._use_jastrow_ee:
      self._jastrow_ee_net = TensorMlp(
          [64, 64, 64, len(self._shape)**2],
          rank=2,
          activate_final=False,
          residual=True)
      # To prevent NaNs, rescale factor by number of electrons squared.
      self._jastrow_ee = self._jastrow_ee_net(
          r_ee, layer_collection=layer_collection) / (
              self._ne**2)

      # split output into channels for different spin interactions
      jastrow_ee = flatten([
          tf.split(x, self._shape, axis=2)
          for x in tf.split(self._jastrow_ee, self._shape, axis=1)
      ])
      # For each spin interaction (alpha/alpha, alpha/beta, beta/alpha,
      # beta/beta), get just the Jastrow term for that interaction
      jastrow_ee = [x[..., i:i + 1] for i, x in enumerate(jastrow_ee)]
      # Add up all Jastrow terms for all pairs of electrons over all spins
      jastrow_ee = tf.add_n([tf.reduce_sum(x, axis=[1, 2]) for x in jastrow_ee])
      total_jastrow += jastrow_ee

    if self._use_jastrow_en:
      self._jastrow_en_net = TensorMlp([64, 64, 64, self._na],
                                       rank=2,
                                       activate_final=False,
                                       residual=True)
      self._jastrow_en = (
          self._jastrow_en_net(
              r_en[..., None], layer_collection=layer_collection) / self._ne /
          self._na)
      # We have one output per atom/electron distance. Rather than having one
      # network for every atom, we have one network with many outputs, and take
      # the diagonal of the outputs.
      jastrow_en = tf.reduce_sum(tf.matrix_diag_part(self._jastrow_en), axis=2)
      jastrow_en = tf.reduce_sum(jastrow_en, axis=1, keepdims=True)
      total_jastrow += jastrow_en

    if self._use_jastrow_een:
      self._jastrow_een_nets = []
      self._jastrow_een = []
      for i in range(self._na):
        # Create a separate network for every atom. Not efficient, but works
        # well enough for the experiments we need this for. This is only for
        # comparison against benchmarks in the literature and is not meant to
        # be a default component of the Fermionic Neural Network.
        self._jastrow_een_nets.append(
            TensorMlp([64, 64, 64, len(self._shape)**2],
                      rank=2,
                      activate_final=False,
                      residual=True))

        batch = tf.concat((r_ee,
                           tf.tile(
                               tf.expand_dims(r_en[..., i:i + 1], axis=1),
                               [1, self._ne, 1, 1]),
                           tf.tile(
                               tf.expand_dims(r_en[..., i:i + 1], axis=2),
                               [1, 1, self._ne, 1])),
                          axis=-1)
        self._jastrow_een.append(self._jastrow_een_nets[i](
            batch, layer_collection=layer_collection) / (self._ne**2))

        # split output into channels for different spin interactions
        jastrow_een = flatten([
            tf.split(x, self._shape, axis=2)
            for x in tf.split(self._jastrow_een[i], self._shape, axis=1)
        ])
        # For each spin interaction (alpha/alpha, alpha/beta, beta/alpha,
        # beta/beta), get just the Jastrow term for that interaction
        jastrow_een = [x[..., i:i + 1] for i, x in enumerate(jastrow_een)]
        # Add up all Jastrow terms for all pairs of electrons over all spins
        jastrow_een = tf.add_n(
            [tf.reduce_sum(x, axis=[1, 2]) for x in jastrow_een])
        total_jastrow += jastrow_een

    if self._output_log:
      output = (output[0] - total_jastrow, output[1])
    else:
      output = output * tf.exp(-total_jastrow)

    self.output = output
    return self.output


def pretrain_hartree_fock(network, data, strategy, hf_approx):
  """Pretrain network so orbitals are nearly uncorrelated.

  Args:
    network: network to be pretrained. Note: only the network._orbitals tensor
      is pretrained.
    data: (nreplicas, batch, spatial_dim * nelectrons) Tensor. Input to network.
    strategy: DistributionStrategy instance.
    hf_approx: Instance of Scf class containing Hartree-Fock approximation.

  Returns:
    hf_loss: Distance between orbitals and Hartree-Fock orbitals. Minimised each
      iteration by tf.AdamOptimizer.
  """

  with strategy.scope():
    optim = tf.train.AdamOptimizer(name='Adam_pretrain')

  def pretrain_step_fn(walkers):
    """Step function for pretraining."""
    replica_id = tf.distribute.get_replica_context().replica_id_in_sync_group
    data = walkers[replica_id]
    network(data)
    xs = tf.reshape(data, [int(data.shape[0]), sum(hf_approx.nelectrons), -1])
    hf_mats = hf_approx.tf_eval_hf(xs, deriv=True)

    # pylint: disable=protected-access
    diffs = [
        tf.expand_dims(hf, 1) - nn
        for (hf, nn) in zip(hf_mats, network._orbitals)
    ]
    # pylint: enable=protected-access
    mses = [tf.reduce_mean(tf.square(diff)) for diff in diffs]
    if len(mses) > 1:
      mse_loss = 0.5 * tf.add(*mses)
    else:
      mse_loss = 0.5 * mses[0]

    pretrain_op = optim.minimize(mse_loss)
    with tf.control_dependencies([pretrain_op]):
      mse_loss = tf.identity(mse_loss)
    return mse_loss

  mse_loss = strategy.experimental_run(
      functools.partial(pretrain_step_fn, walkers=data))
  return strategy.reduce(tf.distribute.ReduceOp.MEAN, mse_loss)
