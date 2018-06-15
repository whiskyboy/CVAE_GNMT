#encoding=utf-8

import math
import jieba
import cPickle as pickle


class Trigram:

    def __init__(self, model_dir):
        print 'initializing tri-gram model.'
        self.model_path = model_dir + '/trigram.pkl'
        self.voc_path = model_dir + '/vocab.txt'

        # help words
        self.unk_word = '<unk>'
        self.frt_word = '<s1>'
        self.scd_word = '<s2>'
        self.lst_word = '</tail>'
        self.appr_zero = 1.0 / 10000

        # load vocabulary
        voc_file = open(self.voc_path, 'r')
        self.w2id = {}
        idx = 0
        for line in voc_file:
            w,_ = line.strip().split('\t')
            self.w2id[w] = idx
            idx += 1
        help_words = [self.frt_word, self.scd_word, self.lst_word]
        for w in help_words:
            self.w2id[w] = idx
            idx += 1
        print('%d words loaded.' % len(self.w2id))

        # load trigram model
        model_file = open(self.model_path, 'rb')
        (self.ug2f, self.bg2f, self.tg2f, 
        self.ug2n, self.ug2n_r, self.bg2n) = pickle.load(model_file)
        model_file.close()
        self.bg_num = len(self.bg2f)
        self.bgD1, self.bgD2, self.bgD3 = self.compute_delta(self.bg2f)
        self.tgD1, self.tgD2, self.tgD3 = self.compute_delta(self.tg2f)
        print 'tri-gram model loaded.'

        ## notification
        print 'Kneser-Ney smoothing used.'
        print('discount for bi-gram,  D1: %.4f, D2: %.4f, D3: %.4f.' % (self.bgD1, self.bgD2, self.bgD3))
        print('discount for tri-gram, D1: %.4f, D2: %.4f, D3: %.4f.' % (self.tgD1, self.tgD2, self.tgD3))

        print 'tri-gram model initialized.'


    def log_probability_sentence(self, sentence):
        """
        Compute the log-probability of the given sentence
            based on tri-gram language model.
        Arg:
            sentence: The given sentence.
        Return:
            The log-probability of the given sentence.
        """
        sentence = ''.join(sentence.split())
        words = jieba.lcut(sentence)
        words = [w.encode('utf-8') for w in words]
        words = [self.frt_word, self.scd_word] + words + [self.lst_word]
        word_ids = [self.word2id(w, self.w2id) for w in words]
        res = 0.0
        for i in range(len(word_ids)-2):
            res += self.log_probability_word_given_two_words(word_ids[i], word_ids[i+1], word_ids[i+2])
        return res/(len(word_ids)-2)

    def log_probability_word_given_two_words(self, widl, widm, widr):
        xijk = 0.0
        xij_ = 0.0
        bg_key = (widl, widm)
        tg_key = (widl, widm, widr)
        if tg_key in self.tg2f:
            xijk = self.tg2f[tg_key]
        if bg_key in self.bg2f:
            xij_ = self.bg2f[bg_key]
        nij_ = 0
        if bg_key in self.bg2n:
            nij_ = self.bg2n[bg_key]
        if   2.0 >  xijk                : delta = self.tgD1
        elif 2.0 <= xijk and 3.0 > xijk : delta = self.tgD2
        elif 3.0 <= xijk                : delta = self.tgD3
        probability_discounted = 0.0
        lmda = 1.0
        if not 0.0 == xij_: 
            probability_discounted = max(xijk-delta, 0) / xij_
            lmda = delta * nij_ / xij_
        res = probability_discounted + lmda * self.probability_word_given_word(widm, widr)
        #assert not 0.0 == res
        if 0.0 == res: res = self.appr_zero
        return math.log(res)

    def probability_word_given_word(self, widl, widr):
        xij = 0.0
        xi_ = 0.0
        ug_key = widl
        bg_key = (widl, widr)
        if bg_key in self.bg2f:
            xij = self.bg2f[bg_key]
        if ug_key in self.ug2f:
            xi_ = self.ug2f[ug_key]
        ni_ = 0
        if ug_key in self.ug2n:
            ni_ = self.ug2n[ug_key]
        if   2.0 >  xij                : delta = self.bgD1
        elif 2.0 <= xij and 3.0 > xij  : delta = self.bgD2
        elif 3.0 <= xij                : delta = self.bgD3
        probability_discounted = 0.0
        lmda = 1.0
        if not 0.0 == xi_:
            probability_discounted = max(xij-delta, 0) / xi_
            lmda = delta * ni_ / xi_
        res = probability_discounted + lmda * self.probability_word(widr)
        return res

    def probability_word(self, wid):
        n_j = 0
        if wid in self.ug2n_r:
            n_j = self.ug2n_r[wid]
        res = float(n_j) / self.bg_num
        #assert not 0.0 == res
        return res

    def word2id(self, word, word_dict):
        if word in word_dict:
            return word_dict[word]
        else:
            return word_dict[self.unk_word]

    def compute_delta(self, ng2f):
        n1 = 0
        n2 = 0
        n3 = 0
        n4 = 0
        for ng in ng2f:
            if 1.0 == ng2f[ng]:
                n1 += 1
            elif 2.0 == ng2f[ng]:
                n2 += 1
            elif 3.0 == ng2f[ng]:
                n3 += 1
            elif 4.0 == ng2f[ng]:
                n4 += 1
        Y = float(n1) / (n1 + 2*n2)
        D1 = 1 - 2*Y*(float(n2) / n1)
        D2 = 2 - 3*Y*(float(n3) / n2)
        D3 = 3 - 4*Y*(float(n4) / n3)
        return D1, D2, D3


