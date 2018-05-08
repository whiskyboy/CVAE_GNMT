#coding:utf8
from __future__ import print_function

import sys
import time
import argparse
import numpy as np
import tensorflow as tf

from .. import cvae_model
from .. import model_helper
from .. import nmt
from ..utils import nmt_utils
from ..utils import misc_utils as utils

utils.check_tensorflow_version()

class AlphaCommentServer(object):
    def __init__(self, model_dir, args=None):
        nmt_parser = argparse.ArgumentParser()
        nmt.add_arguments(nmt_parser)
        FLAGS, _ = nmt_parser.parse_known_args(args)
        default_hparams = nmt.create_hparams(FLAGS)
        self.hparams = nmt.create_or_load_hparams(model_dir, default_hparams, FLAGS.hparams_path, save_hparams=False)
        self.hparams.beam_width = 0 # force use greedy decoder for inference
        self.ckpt = tf.train.latest_checkpoint(model_dir)
        self.infer_model = model_helper.create_infer_model(cvae_model.CVAEModel, self.hparams)
        self.sess = tf.Session(graph=self.infer_model.graph, config=utils.get_config_proto())
        self.loaded_infer_model = model_helper.load_model(
                self.infer_model.model, self.ckpt, self.sess, "infer")

    def comment(self, title, sample_num=30, batch_size=32):
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
        num_sentences = 0
        comment_list = []
        while True:
            try:
                nmt_outputs, _ = self.loaded_infer_model.decode(self.sess)
                if self.hparams.beam_width == 0:
                    nmt_outputs = np.expand_dims(nmt_outputs, 0)

                batch_size = nmt_outputs.shape[1]
                num_sentences += batch_size

                for sent_id in range(batch_size):
                    translation = nmt_utils.get_translation(
                        nmt_outputs[0],
                        sent_id,
                        tgt_eos=self.hparams.eos,
                        subword_option=self.hparams.subword_option)
                    comment_list.append(translation)
            except tf.errors.OutOfRangeError:
                utils.print_time(
                    "  done, num of outputs %d" % num_sentences, start_time)
                break
        return comment_list

if __name__=="__main__":
    server = AlphaCommentServer(model_dir=sys.argv[1])
    title = "朱婷 实现 mvp 俱乐部 大满贯 ， 昔日 对手 ： 唯一 佩服 的 就是 中国 婷 ！"
    for comment in server.comment(title):
        print(comment)
