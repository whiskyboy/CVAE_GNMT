import sys
from flask import *
from ..serving.inference_service import AlphaCommentServer

app = Flask(__name__)
comment_server = AlphaCommentServer(model_dir=sys.argv[1])

@app.route("/AlphaComment", methods=['POST'])
def GetComment():
    req_json = request.get_json()
    if "title" not in req_json:
        return jsonify({"Error": "Bad Request"}), 403
    title = req_json["title"]
    return jsonify(comment_server.comment(title))


if __name__ == "__main__":
    app.run(host='0.0.0.0')