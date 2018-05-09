#coding:utf8
import sys
from serving.inference_service import AlphaCommentServer

if __name__=="__main__":
    server = AlphaCommentServer(model_dir=sys.argv[1])
    title = "朱婷 实现 mvp 俱乐部 大满贯 ， 昔日 对手 ： 唯一 佩服 的 就是 中国 婷 ！"
    for comment in server.comment(title):
        print(comment)
    title = "总结 ： 长寿 的 人 ， 一般 都 坚持 九大 养身 习惯 ， 你 做 对 几个 ？"
    for comment in server.comment(title):
        print(comment)
    title = "结婚 婆婆 给 彩礼 18 万 ， 我家 陪嫁 一辆车"
    for comment in server.comment(title):
        print(comment)
