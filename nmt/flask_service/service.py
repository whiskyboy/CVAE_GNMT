import os
from flask import *
import jieba
import json
import argparse
from ..serving.inference_service import AlphaCommentServer

app = Flask(__name__)

def add_arguments(parser):
    parser.add_argument("--cvae_model_dir", type=str, required=True,
                        help="path of cvae model")
    parser.add_argument("--lm_model_dir", type=str, required=True,
                        help="path of language model")
    parser.add_argument("--sample_num", type=int, default=80,
                        help="sample number for generation")
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

def preprocess_text(query):
    return " ".join(reformat_tokens(jieba.lcut(query), src_vocabs)).encode("utf8")

def pack_response(query, responses):
    response_list = []
    for response, score in responses:
        response_list.append({
            "Response": response,
            "Score": score
        })
    return {
        "Query": query,
        "ResponseList": response_list
    }


@app.route("/AlphaComment", methods=['POST'])
def GetComment():
    req_json = request.get_json()
    print "Query: %s"%json.dumps(req_json)
    if req_json is None or "Query" not in req_json:
        return jsonify({"Error": "Bad Request"}), 403
    query = preprocess_text(req_json["Query"])
    ppl_cutoff_score = req_json.get("PPLCutoffScore", -3.0)
    lm_cutoff_score = req_json.get("LMCutoffScore", -5.0)
    pmi_cutoff_score = req_json.get("PMICutoffScore", 1.0)
    max_return = req_json.get("MaxReturn", 10)
    min_response_len = req_json.get("MinResponseLen", 3)
    max_response_len = req_json.get("MaxResponseLen", 20)
    
    responses = comment_server.comment(query)
    valid_responses = []
    for comment, (ppl, lm_prob, lm_prob_details, pmi) in responses:
        if len(comment.split(" ")) < min_response_len or \
           len(comment.split(" ")) > max_response_len:
            continue

        if ppl <= ppl_cutoff_score:
            continue

        if lm_prob <= lm_cutoff_score:
            continue

        if pmi <= pmi_cutoff_score:
            continue

        isValid = True
        for p in lm_prob_details:
            if p <= lm_cutoff_score * 2:
                isValid = False
                break
        if not isValid:
            continue

        valid_responses.append((comment, (ppl, lm_prob, pmi)))

    if max_return > 0:
        responses = sorted(valid_responses, key=lambda x: x[1][2], reverse=True)[:max_return]
    else:
        responses = sorted(valid_responses, key=lambda x: x[1][2], reverse=True)
    
    return jsonify(pack_response(query, responses))


if __name__ == "__main__":
    app_parser = argparse.ArgumentParser()
    add_arguments(app_parser)
    FLAGS, _ = app_parser.parse_known_args()

    if FLAGS.src_vocab_file is None:
        FLAGS.src_vocab_file = os.path.join(FLAGS.cvae_model_dir, "vocab.in")
    if FLAGS.tgt_vocab_file is None:
        FLAGS.tgt_vocab_file = os.path.join(FLAGS.cvae_model_dir, "vocab.out")

    comment_server = AlphaCommentServer(cvae_model_dir=FLAGS.cvae_model_dir,
                                        lm_model_dir=FLAGS.lm_model_dir,
                                        sample_num=FLAGS.sample_num,
                                        src_vocab_file=FLAGS.src_vocab_file,
                                        tgt_vocab_file=FLAGS.tgt_vocab_file)

    src_vocabs = load_vocab_file(FLAGS.src_vocab_file)
    tgt_vocabs = load_vocab_file(FLAGS.tgt_vocab_file)

    app.run(host='0.0.0.0', port=80, threaded=True)
