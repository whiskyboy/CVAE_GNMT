# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CVAE sequence-to-sequence model with GNMT support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# TODO(rzhao): Use tf.contrib.framework.nest once 1.3 is out.
from tensorflow.python.util import nest

from . import gnmt_model
from . import model_helper
from .utils import misc_utils as utils

__all__ = ["CVAEModel"]


class CVAEModel(gnmt_model.GNMTModel):
  """Sequence-to-sequence dynamic model with CVAE architecture.
  """

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):
    super(CVAEModel, self).__init__(
        hparams=hparams,
        mode=mode,
        iterator=iterator,
        source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope,
        extra_args=extra_args)

  def _build_encoder(self, hparams):
    """Build a CVAE encoder."""

    # if hparams.encoder_type == "uni" or hparams.encoder_type == "bi":
    #   return super(CVAEModel, self)._build_encoder(hparams)
    #
    # if hparams.encoder_type != "gnmt":
    #   raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

    # Build CVAE encoder.
    num_bi_layers = 1
    num_uni_layers = self.num_encoder_layers - num_bi_layers
    utils.print_out("  num_bi_layers = %d" % num_bi_layers)
    utils.print_out("  num_uni_layers = %d" % num_uni_layers)

    iterator = self.iterator

    self.cvae_latent_size = hparams.cvae_latent_size
    self.bow_latent_size = hparams.bow_latent_size
    self.full_kl_step = hparams.full_kl_step

    with tf.variable_scope("src_encoder") as scope:
      source = iterator.source
      if self.time_major:
        source = tf.transpose(source)
      src_encoder_outputs, src_encoder_state = self._get_sequence_encoder(source, iterator.source_sequence_length,
                                                                          self.embedding_encoder,
                                                                          num_uni_layers, num_bi_layers,
                                                                          hparams, scope)
      encoder_outputs = src_encoder_outputs

    with tf.variable_scope("tgt_encoder") as scope:
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        target = iterator.target
        if self.time_major:
          target = tf.transpose(target)
        tgt_encoder_outputs, tgt_encoder_state = self._get_sequence_encoder(target, iterator.target_sequence_length-1,
                                                                           self.embedding_decoder,
                                                                           num_uni_layers, num_bi_layers,
                                                                           hparams, scope)

    _num_encoder_layers = len(src_encoder_state)
    self.decoder_init_state = [0] * _num_encoder_layers
    self.prior_mu = [0] * _num_encoder_layers
    self.prior_logvar = [0] * _num_encoder_layers
    self.recog_mu = [0] * _num_encoder_layers
    self.recog_logvar = [0] * _num_encoder_layers
    for i in range(_num_encoder_layers):
      _num_rnn_layers = len(src_encoder_state[i])
      self.decoder_init_state[i] = [0] * _num_rnn_layers
      self.prior_mu[i] = [0] * _num_rnn_layers
      self.prior_logvar[i] = [0] * _num_rnn_layers
      self.recog_mu[i] = [0] * _num_rnn_layers
      self.recog_logvar[i] = [0] * _num_rnn_layers
      for j in range(_num_rnn_layers):
        _src_state = src_encoder_state[i][j]
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
          _tgt_state = tgt_encoder_state[i][j]
        else:
          _tgt_state = None
        _init_state, _prior_mu, _prior_logvar, _recog_mu, _recog_logvar = \
          self._get_decoder_init_state(_src_state, _tgt_state, "%s_%s"%(i,j), hparams)
        self.decoder_init_state[i][j] = _init_state
        self.prior_mu[i][j] = _prior_mu
        self.prior_logvar[i][j] = _prior_logvar
        self.recog_mu[i][j] = _recog_mu
        self.recog_logvar[i][j] = _recog_logvar
      self.decoder_init_state[i] = tf.contrib.rnn.LSTMStateTuple(*self.decoder_init_state[i])
      self.prior_mu[i] = tf.contrib.rnn.LSTMStateTuple(*self.prior_mu[i])
      self.prior_logvar[i] = tf.contrib.rnn.LSTMStateTuple(*self.prior_logvar[i])
      self.recog_mu[i] = tf.contrib.rnn.LSTMStateTuple(*self.recog_mu[i])
      self.recog_logvar[i] = tf.contrib.rnn.LSTMStateTuple(*self.recog_logvar[i])
    self.decoder_init_state = tuple(self.decoder_init_state)
    self.prior_mu = tuple(self.prior_mu)
    self.prior_logvar = tuple(self.prior_logvar)
    self.recog_mu = tuple(self.recog_mu)
    self.recog_logvar = tuple(self.recog_logvar)

    return encoder_outputs, self.decoder_init_state

  def _get_decoder_init_state(self, src_state, tgt_state, scope_idx, hparams):
    with tf.variable_scope("priorNetwork_%s" % scope_idx) as scope:
      prior_mulogvar = tf.contrib.layers.fully_connected(src_state, self.cvae_latent_size * 2)
      prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=-1)
      latent_sample = self._sample_gaussian(prior_mu, prior_logvar)

    with tf.variable_scope("recogNetwork_%s" % scope_idx) as scope:
      if tgt_state is not None:
        recog_input = tf.concat([src_state, tgt_state], -1)
        recog_mulogvar = tf.contrib.layers.fully_connected(recog_input, self.cvae_latent_size * 2)
        recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=-1)
        latent_sample = self._sample_gaussian(recog_mu, recog_logvar)
      else:
        recog_mu, recog_logvar = None, None

    with tf.variable_scope("generationNetwork_%s" % scope_idx) as scope:
      dec_inputs = tf.concat([src_state, latent_sample], -1)
      decoder_init_state = tf.contrib.layers.fully_connected(dec_inputs, hparams.num_units)

    return decoder_init_state, prior_mu, prior_logvar, recog_mu, recog_logvar

  def _compute_loss(self, logits):
    decoder_loss = super(CVAEModel, self)._compute_loss(logits)
    bow_loss = self._compute_bow_loss()
    kl_loss = self._compute_kl_loss()
    return decoder_loss + bow_loss + kl_loss

  def _compute_bow_loss(self):
    labels = self.iterator.target
    if self.time_major:
      labels = tf.transpose(labels)
    max_time = self.get_max_time(labels)

    with tf.variable_scope("bow_decoder") as scope:
      bow_init_state = self._multi_tensor_concat(self.decoder_init_state)
      bow_fc = tf.contrib.layers.fully_connected(bow_init_state, self.bow_latent_size, activation_fn=tf.tanh)
      bow_logits = tf.contrib.layers.fully_connected(bow_fc, self.tgt_vocab_size)
      if self.time_major:
        tile_bow_logits = tf.tile(tf.expand_dims(bow_logits, 0), [max_time, 1, 1])
      else:
        tile_bow_logits = tf.tile(tf.expand_dims(bow_logits, 1), [1, max_time, 1])

    label_mask = tf.sequence_mask(self.iterator.target_sequence_length - 1, max_time, dtype=tile_bow_logits.dtype)
    if self.time_major:
      label_mask = tf.transpose(label_mask)

    bow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tile_bow_logits, labels=labels) * label_mask
    mean_bow_loss = tf.reduce_sum(bow_loss) / tf.to_float(self.batch_size)

    return mean_bow_loss

  def _compute_kl_loss(self):
    prior_mu = self._multi_tensor_concat(self.prior_mu)
    prior_logvar = self._multi_tensor_concat(self.prior_logvar)
    recog_mu = self._multi_tensor_concat(self.recog_mu)
    recog_logvar = self._multi_tensor_concat(self.recog_logvar)
    kl_loss = self._gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
    mean_kl_loss = tf.reduce_sum(kl_loss) / tf.to_float(self.batch_size)
    kl_weights = tf.minimum(tf.to_float(self.global_step) / self.full_kl_step, 1.0)

    return kl_weights * mean_kl_loss

  def _get_sequence_encoder(self, sequence, sequence_length, embedding_table,
                            num_uni_layers, num_bi_layers, hparams, scope):
    dtype = scope.dtype

    # Look up embedding, emp_inp: [max_time, batch_size, num_units]
    #   when time_major = True
    encoder_emb_inp = tf.nn.embedding_lookup(embedding_table,
                                             sequence)

    # Execute _build_bidirectional_rnn from Model class
    bi_encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
        inputs=encoder_emb_inp,
        sequence_length=sequence_length,
        dtype=dtype,
        hparams=hparams,
        num_bi_layers=num_bi_layers,
        num_bi_residual_layers=0,  # no residual connection
    )

    uni_cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=num_uni_layers,
        num_residual_layers=self.num_encoder_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=self.num_gpus,
        base_gpu=1,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    # encoder_outputs: size [max_time, batch_size, num_units]
    #   when time_major = True
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        uni_cell,
        bi_encoder_outputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=self.time_major)

    # Pass all encoder state except the first bi-directional layer's state to
    # decoder.
    encoder_state = (bi_encoder_state[1],) + (
        (encoder_state,) if num_uni_layers == 1 else encoder_state)

    return encoder_outputs, encoder_state

  def _sample_gaussian(self, mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z = mu + tf.multiply(std, epsilon)
    return z

  def _gaussian_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar),
                                        tf.exp(prior_logvar)),
                               reduction_indices=-1)
    return kld

  def _multi_tensor_concat(self, ternsor_list):
    bi_tensor = tf.concat(ternsor_list, -1)
    cat_tensor = tf.concat([bi_tensor[0], bi_tensor[1]], -1)

    return cat_tensor
