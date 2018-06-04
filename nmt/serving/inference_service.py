#coding:utf8
from __future__ import print_function

import os
import time
import argparse
from itertools import groupby
import numpy as np
import tensorflow as tf

from .. import cvae_model
from .. import model_helper
from .. import nmt
from ..utils import nmt_utils
from ..utils import misc_utils as utils

utils.check_tensorflow_version()

class AlphaCommentServer(object):
    def __init__(self, model_dir, src_vocab_file=None, tgt_vocab_file=None, args=None):
        nmt_parser = argparse.ArgumentParser()
        nmt.add_arguments(nmt_parser)
        FLAGS, _ = nmt_parser.parse_known_args(args)
        default_hparams = nmt.create_hparams(FLAGS)
        self.hparams = nmt.create_or_load_hparams(model_dir, default_hparams, FLAGS.hparams_path, save_hparams=False)
        self.hparams.beam_width = 0 # force use greedy decoder for inference
        if src_vocab_file:
            self.hparams.src_vocab_file = src_vocab_file
        else:
            self.hparams.src_vocab_file = os.path.join(model_dir, "vocab.in")
        if tgt_vocab_file:
            self.hparams.tgt_vocab_file = tgt_vocab_file
        else:
            self.hparams.tgt_vocab_file = os.path.join(model_dir, "vocab.out")
        self.ckpt = tf.train.latest_checkpoint(model_dir)
        self.infer_model = model_helper.create_infer_model(cvae_model.CVAEModel, self.hparams)
        self.sess = tf.Session(graph=self.infer_model.graph, config=utils.get_config_proto())
        with self.infer_model.graph.as_default():
            self.loaded_infer_model = model_helper.load_model(
                self.infer_model.model, self.ckpt, self.sess, "infer")

    def _refineAndValidateComment(self, comment):
        tokens = comment.split()
        if "<unk>" in tokens:
            return None
        refined_tokens = [k for k, g in groupby(tokens)]  # remove consecutive duplicated tokens
        if len(refined_tokens) != len(set(refined_tokens)):  # still has non-consecutive duplicated tokens
            return None
        return " ".join(refined_tokens)


    def comment(self, title, sample_num=30, batch_size=30, lm_score=False):
        if batch_size > sample_num:
            batch_size = sample_num

        infer_data = [title] * sample_num
        self.sess.run(
            self.infer_model.iterator.initializer,
            feed_dict={
                self.infer_model.src_placeholder: infer_data,
                self.infer_model.batch_size_placeholder: batch_size
            })

        # Decode
        utils.print_out("# Start decoding with title: %s" % title)

        start_time = time.time()
        comments = {}
        while True:
            try:
                if lm_score:
                    nmt_outputs, nmt_logp = self.loaded_infer_model.decode_with_logp(self.sess)
                else:
                    nmt_outputs, _ = self.loaded_infer_model.decode(self.sess)
                    nmt_logp = None
                
                if self.hparams.beam_width > 0:
                    nmt_outputs = nmt_outputs[0]
                    if nmt_logp is not None:
                        nmt_logp = nmt_logp[0]

                batch_size = nmt_outputs.shape[0]
                for sent_id in range(batch_size):
                    translation, score = nmt_utils.get_translation_with_score(
                        nmt_outputs,
                        nmt_logp,
                        sent_id,
                        tgt_eos=self.hparams.eos,
                        subword_option=self.hparams.subword_option)
                    utils.print_out("sample comment: %s lm score: %s"%(translation, score))
                    refined_trans = self._refineAndValidateComment(translation)
                    if refined_trans:
                        utils.print_out("refined comment: %s"%refined_trans)
                        comments[refined_trans] = score
            except tf.errors.OutOfRangeError:
                utils.print_time(
                    "  done, num of outputs %d"%len(comments), start_time)
                break
        return sorted(comments.items(), key=lambda x: x[1])
