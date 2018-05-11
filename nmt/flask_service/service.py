import os
from flask import *
import jieba
import argparse
from ..serving.inference_service import AlphaCommentServer

app = Flask(__name__)

def add_arguments(parser):
    parser.add_argument("--model_dir", type=str, required=True,
                        help="path of model for comment generation")
    parser.add_argument("--src_vocab_file", type=str, default=None,
                        help="path of source(title) vocabulary file")
    parser.add_argument("--tgt_vocab_file", type=str, default=None,
                        help="path of target(comment) vocabulary file")

def load_vocab_file(vocab_file):
    vocabs = set()
    with open(vocab_file, 'r') as fin:
        for line in fin:
            vocabs.add(line.strip().decode("utf8"))
    return vocabs

def reformat_tokens(tokens, vocabs):
    ret_tokens = []
    for token in tokens:
        if token not in vocabs:
            ret_tokens.extend(token)
        else:
            ret_tokens.append(token)
    return ret_tokens

def preprocess_text(title):
    return " ".join(reformat_tokens(jieba.lcut(title), src_vocabs)).encode("utf8")

def pack_response(title, comments):
    comment_list = []
    for comment, score in comments:
        comment_list.append({
            "Comment": comment,
            "Score": score
        })
    return {
        "Title": title,
        "CommentList": comment_list
    }


@app.route("/AlphaComment", methods=['POST'])
def GetComment():
    req_json = request.get_json()
    if req_json is None or "Title" not in req_json:
        return jsonify({"Error": "Bad Request"}), 403
    title = preprocess_text(req_json["Title"])
    sample_num = req_json.get("SampleNum", 30)
    batch_size = req_json.get("BatchSize", 30)
    lm_score = req_json.get("LMScore", False)
    comments = comment_server.comment(title, sample_num, batch_size, lm_score)
    return jsonify(pack_response(title, comments))


if __name__ == "__main__":
    app_parser = argparse.ArgumentParser()
    add_arguments(app_parser)
    FLAGS, _ = app_parser.parse_known_args()

    if FLAGS.src_vocab_file is None:
        FLAGS.src_vocab_file = os.path.join(FLAGS.model_dir, "vocab.in")
    if FLAGS.tgt_vocab_file is None:
        FLAGS.tgt_vocab_file = os.path.join(FLAGS.model_dir, "vocab.out")

    comment_server = AlphaCommentServer(model_dir=FLAGS.model_dir,
                                        src_vocab_file=FLAGS.src_vocab_file,
                                        tgt_vocab_file=FLAGS.tgt_vocab_file)

    src_vocabs = load_vocab_file(FLAGS.src_vocab_file)
    tgt_vocabs = load_vocab_file(FLAGS.tgt_vocab_file)

    app.run(host='0.0.0.0', port=80)
