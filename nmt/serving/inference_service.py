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
from ..trigram_tool import trigram

utils.check_tensorflow_version()

PMI_LAMBDA = 1.0

class AlphaCommentServer(object):
    def __init__(self, cvae_model_dir, lm_model_dir, sample_num=80, src_vocab_file=None, tgt_vocab_file=None, args=None):
        nmt_parser = argparse.ArgumentParser()
        nmt.add_arguments(nmt_parser)
        FLAGS, _ = nmt_parser.parse_known_args(args)
        default_hparams = nmt.create_hparams(FLAGS)
        self.hparams = nmt.create_or_load_hparams(cvae_model_dir, default_hparams, FLAGS.hparams_path, save_hparams=False)
        self.hparams.sample_num = sample_num # for inference
        self.hparams.beam_width = 0 # force use greedy decoder for inference
        if src_vocab_file:
            self.hparams.src_vocab_file = src_vocab_file
        else:
            self.hparams.src_vocab_file = os.path.join(cvae_model_dir, "vocab.in")
        if tgt_vocab_file:
            self.hparams.tgt_vocab_file = tgt_vocab_file
        else:
            self.hparams.tgt_vocab_file = os.path.join(cvae_model_dir, "vocab.out")
        self.ckpt = tf.train.latest_checkpoint(cvae_model_dir)
        self.infer_model = model_helper.create_infer_model(cvae_model.CVAEModel, self.hparams)
        self.sess = tf.Session(graph=self.infer_model.graph, config=utils.get_config_proto())
        with self.infer_model.graph.as_default():
            self.loaded_infer_model = model_helper.load_model(
                self.infer_model.model, self.ckpt, self.sess, "infer")
        # load lm model of ngram
        self.lm_model = trigram.Trigram(lm_model_dir)

    def refine(self, comment):
        tokens = comment.split()
        if "<unk>" in tokens:
            return None
        refined_tokens = [k for k, g in groupby(tokens)]  # remove consecutive duplicated tokens
        if len(refined_tokens) != len(set(refined_tokens)):  # still has non-consecutive duplicated tokens
            return None
        return " ".join(refined_tokens)


    def comment(self, title):
        infer_data = [title]
        self.sess.run(
            self.infer_model.iterator.initializer,
            feed_dict={
                self.infer_model.src_placeholder: infer_data,
                self.infer_model.batch_size_placeholder: 1,
            })

        # Decode
        utils.print_out("# Start decoding with title: %s" % title)

        start_time = time.time()
        comments = {}
        while True:
            try:
                nmt_outputs, nmt_logp = self.loaded_infer_model.decode_with_logp(self.sess)
                
                if self.hparams.beam_width > 0:
                    nmt_outputs = nmt_outputs[0]
                    nmt_logp = nmt_logp[0]

                _sample_num = nmt_outputs.shape[0]
                for sent_id in range(_sample_num):
                    translation, ppl = nmt_utils.get_translation_with_ppl(
                        nmt_outputs,
                        nmt_logp,
                        sent_id,
                        tgt_eos=self.hparams.eos,
                        subword_option=self.hparams.subword_option)
                    refined_trans = self.refine(translation)
                    if refined_trans:
                        lm_prob, lm_prob_details = self.lm_model.log_probability_sentence(refined_trans)
                        pmi = np.abs(ppl - lm_prob * PMI_LAMBDA)
                        comments[refined_trans] = (ppl, lm_prob, lm_prob_details, pmi)
                        utils.print_out("sample comment: %s ppl: %s lm_prob: %s pmi: %s"%(refined_trans, ppl, lm_prob, pmi))
            except tf.errors.OutOfRangeError:
                utils.print_time(
                    "  done, num of outputs %d"%len(comments), start_time)
                break
        return comments.items()
