#coding:utf8
import sys
from tqdm import tqdm
import requests

def getGenerativeCommentList(query):
    host = "http://0.0.0.0:80/AlphaComment"
    data = {
        "Title": query.strip(),
        "SampleNum": 80,
        "BatchSize": 80,
        "LMScore": True
    }
    response = requests.post(host, json=data)
    if response:
        response = response.json()
        if response.has_key("CommentList"):
            commentList = []
            for detail in response["CommentList"]:
                comment = detail["Comment"]
                score = detail["Score"]
                commentList.append((comment.encode("utf8"), float(score)))
            return commentList

if __name__=="__main__":
    test_file = sys.argv[1]
    output_file = sys.argv[2]

    dataset = []
    with open(test_file, 'r') as fin:
        for line in fin:
            dataset.append(line.strip())

    with open(output_file, 'w') as fout:
        for query in tqdm(dataset):
            comment_list = getGenerativeCommentList(query)
            if comment_list:
                for comment, score in comment_list:
                    fout.write("%s\t%s\t%s\n"%(query, comment, score))
