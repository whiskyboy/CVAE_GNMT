# encoding=utf-8

import os
import sys
import math
import jieba
import logging
import cPickle as pickle

api_methods = [
    'log_probability_sentence',
    'log_probability_whether_sentence_ends'
]

class Trigram:

    def __init__(self, model_dir):
        print 'start initializing tri-gram model.'
        self.model_path = model_dir + '/trigram.pkl'
        self.voc_path = model_dir + '/vocab.txt'

        # debug
        #self.err_file = open('error.txt', 'w')

        # help words
        self.unk_word = '<unk>'
        self.frt_word = '<s1>'
        self.scd_word = '<s2>'
        self.lst_word = '</tail>'
        self.appr_zero = 1.0 / 10000

        # load vocabulary
        voc_file = open(self.voc_path, 'r')
        self.w2id = {}
        self.id2w = {}
        idx = 0
        for line in voc_file:
            w,_ = line.strip().split('\t')
            self.w2id[w] = idx
            #self.id2w[idx] = w # debug
            idx += 1
        help_words = [self.frt_word, self.scd_word, self.lst_word]
        for w in help_words:
            self.w2id[w] = idx
            #self.id2w[idx] = w # debug
            idx += 1
        print('%d words loaded.' % len(self.w2id))

        # load trigram model
        print 'loading tri-grams...'
        model_file = open(self.model_path, 'rb')
        self.tg2f = pickle.load(model_file)
        model_file.close()
        print('%d tri-grams loaded.' % (len(self.tg2f)))

        # compute help variables
        self.__compute_help_variables()
        self.bg_num = len(self.bg2f)
        print 'tri-gram model loaded.'

    def log_probability_sentence(self, sentence):
        """
        Compute the averaged-log-probability of the 
        given sentence based on tri-gram language model.
        The sentence will be re-cutted using jieba.
        Arg:
            sentence: The given sentence.
        Return:
            log(p(sentence)) / words_num
        """
        word_ids = self.__preproc_sentence(sentence)
        res = 0.0
        for i in range(len(word_ids)-2):
            res += self.__log_probability_word_given_two_words(word_ids[i], word_ids[i+1], word_ids[i+2])
        return res/(len(word_ids)-2)

        
    def log_probability_whether_sentence_ends(self, sentence):
        """
        Compute the log-probability of the end of a sentence.
        The sentence will be re-cutted using jieba.
        Arg:
            sentence: The given sentence.
        Return: 
            the log-probability of sentence ending.
        """
        word_ids = self.__preproc_sentence(sentence)
        res = self.__log_probability_word_given_two_words(word_ids[-3], word_ids[-2], word_ids[-1])
        return res


    def __preproc_sentence(self, sentence):
        """
        Recut the sentence, add help words(<s1>,<s2>,</tail>), and transform to word ids.
        Arg: 
            sentence: the given sentence.
        Return:
            word ids
        """
        sentence = ''.join(sentence.split())
        words = jieba.lcut(sentence)
        words = [w.encode('utf-8') for w in words]
        words = [self.frt_word, self.scd_word] + words + [self.lst_word]
        word_ids = [self.__word2id(w, self.w2id) for w in words]
        return word_ids

    def __compute_help_variables(self):
        """ 1. bi-gram and uni-gram
            2. bi-gram/uni-gram following word types num
            3. D1, D2, D3 used in smoothing
        """
        print '\ncomputing help variables...'
        def upd_once_ug2n(key_wid, wid, ug2ws):
            if key_wid in ug2ws:
                ug2ws[key_wid].add(wid)
            else:
                temp_set = set([wid])
                ug2ws[key_wid] = temp_set

        def upd_once_bg2n(key, wid, bg2ws):
            if key in bg2ws:
                bg2ws[key].add(wid)
            else:
                temp_set = set([wid])
                bg2ws[key] = temp_set

        def trans_set_to_num(ng2ws):
            ng2n = {}
            for k in ng2ws:
                ng2n[k] = len(ng2ws[k])
            return ng2n

        def compute_discount(n1, n2, n3, n4):
            Y = float(n1) / (n1 + 2*n2)
            D1 = 1 - 2*Y*(float(n2) / n1)
            D2 = 2 - 3*Y*(float(n3) / n2)
            D3 = 3 - 4*Y*(float(n4) / n3)
            return D1, D2, D3

        print 'itering tri-gram...'
        bg2f = {}
        tg2f = self.tg2f
        bg2ws_once, bg2ws_twice, bg2ws_other = {}, {}, {}
        n1, n2, n3, n4 = 0, 0, 0, 0
        idx = 0
        for k in tg2f:
            widl, widm, widr = k
            freq = tg2f[k]
            bg = (widl, widm)
            if bg in bg2f:  bg2f[bg] += tg2f[k]
            else:           bg2f[bg] = tg2f[k]
    
            if 1 == freq:   
                upd_once_bg2n(bg, widr, bg2ws_once)
                n1 += 1
            elif 2 == freq: 
                upd_once_bg2n(bg, widr, bg2ws_twice)
                n2 += 1
            else:           
                upd_once_bg2n(bg, widr, bg2ws_other)
                if 3 == freq: n3 += 1
                elif 4 == freq: n4 += 1
            idx += 1
            if 0 == idx%10000:
                sys.stdout.write('%dw/%dw processd\r' % (idx/10000, len(tg2f)/10000))
                sys.stdout.flush()
        print
        self.bg2f = bg2f
        self.bg2n_once  = trans_set_to_num(bg2ws_once)
        self.bg2n_twice = trans_set_to_num(bg2ws_twice)
        self.bg2n_other = trans_set_to_num(bg2ws_other)
        self.tgD1, self.tgD2, self.tgD3 = compute_discount(n1, n2, n3, n4)
        print('%d bi-grams got.' % len(self.bg2f))
        print('bi-gram following-word types num got.')
        print('discount for tri-gram, D1: %.4f, D2: %.4f, D3: %.4f.' % (self.tgD1, self.tgD2, self.tgD3))

        print '\nitering bi-gram...'
        ug2f = {}
        bg2f = self.bg2f
        ug2ws_once, ug2ws_twice, ug2ws_other = {}, {}, {}
        ug2ws_r = {}
        n1, n2, n3, n4 = 0, 0, 0, 0
        idx = 0
        for k in bg2f:
            widl, widr = k
            freq = bg2f[k]
            ug = widl
            if ug in ug2f:  ug2f[ug] += bg2f[k]
            else:           ug2f[ug] = bg2f[k]

            if 1 == freq:   
                upd_once_ug2n(widl, widr, ug2ws_once)
                n1 += 1
            elif 2 == freq: 
                upd_once_ug2n(widl, widr, ug2ws_twice)
                n2 += 1
            else:           
                upd_once_ug2n(widl, widr, ug2ws_other)
                if 3 == freq: n3 += 1
                elif 4 == freq: n4 += 1
            upd_once_ug2n(widr, widl, ug2ws_r)
            idx += 1
            if 0 == idx%10000:
                sys.stdout.write('%dw/%dw processd\r' % (idx/10000, len(bg2f)/10000))
                sys.stdout.flush()
        print
        self.ug2f = ug2f
        self.ug2n_once = trans_set_to_num(ug2ws_once)
        self.ug2n_twice = trans_set_to_num(ug2ws_twice)
        self.ug2n_other = trans_set_to_num(ug2ws_other)
        self.ug2n_r = trans_set_to_num(ug2ws_r)
        self.bgD1, self.bgD2, self.bgD3 = compute_discount(n1, n2, n3, n4)
        print('%d uni-grams got.' % len(self.ug2f))
        print('uni-gram following-word types num got.')
        print('discount for bi-gram,  D1: %.4f, D2: %.4f, D3: %.4f.' % (self.bgD1, self.bgD2, self.bgD3))


    def __log_probability_word_given_two_words(self, widl, widm, widr):
        bg_key = (widl, widm)
        tg_key = (widl, widm, widr)

        xijk, xij_ = 0.0, 0.0
        if tg_key in self.tg2f:
            xijk = self.tg2f[tg_key]
        if bg_key in self.bg2f:
            xij_ = self.bg2f[bg_key]

        nij_once, nij_twice, nij_other = 0, 0, 0
        if bg_key in self.bg2n_once:
            nij_once = self.bg2n_once[bg_key]
        if bg_key in self.bg2n_twice:
            nij_twice = self.bg2n_twice[bg_key]
        if bg_key in self.bg2n_other:
            nij_other = self.bg2n_other[bg_key]

        delta = 0.0
        if   0 == xijk : pass
        elif 1 == xijk : delta = self.tgD1
        elif 2 == xijk : delta = self.tgD2
        else :           delta = self.tgD3
        probability_discounted = 0.0
        lmda = 1.0
        if not 0.0 == xij_: 
            probability_discounted = (xijk-delta) / xij_
            lmda = (self.tgD1*nij_once + self.tgD2*nij_twice + self.tgD3*nij_other) / xij_
        res = probability_discounted + lmda * self.__probability_word_given_word(widm, widr)
        #try: # debug
        #    assert res <= 1.0
        #except AssertionError, e:
        #    err_line =  ('w1 %s, w2 %s, w3 %s, xijk %f, xij_ %f, delta %f, probability_discounted %f, nij_once %f, nij_twice %f, nij_other %f, lmda %f, res_prob %f' % 
        #                (self.id2w[widl], self.id2w[widm], self.id2w[widr], xijk, xij_, delta, probability_discounted, nij_once, nij_twice, nij_other, lmda, res) )
        #    self.err_file.write(err_line+'\n')
        #    print err_line
        #assert not 0.0 == res
        if 0.0 == res: res = self.appr_zero
        return math.log(res)

    def __probability_word_given_word(self, widl, widr):
        ug_key = widl
        bg_key = (widl, widr)

        xij = 0.0
        xi_ = 0.0
        if bg_key in self.bg2f:
            xij = self.bg2f[bg_key]
        if ug_key in self.ug2f:
            xi_ = self.ug2f[ug_key]

        ni_once, ni_twice, ni_other = 0, 0, 0
        if ug_key in self.ug2n_once:
            ni_once = self.ug2n_once[ug_key]
        if ug_key in self.ug2n_twice:
            ni_twice = self.ug2n_twice[ug_key]
        if ug_key in self.ug2n_other:
            ni_other = self.ug2n_other[ug_key]

        delta = 0.0
        if 0 == xij :     pass
        elif 1.0 == xij : delta = self.bgD1
        elif 2.0 == xij : delta = self.bgD2
        else :            delta = self.bgD3
        probability_discounted = 0.0
        lmda = 1.0
        if not 0.0 == xi_:
            probability_discounted = (xij-delta) / xi_
            lmda = (self.bgD1*ni_once + self.bgD2*ni_twice + self.bgD3*ni_other) / xi_
        res = probability_discounted + lmda * self.__probability_word(widr)
        #try: # debug
        #    assert res <= 1.0
        #except AssertionError, e:
        #    err_line =  ('w1 %s, w2 %s, xij %f, xi_ %f, delta %f, probability_discounted %f, ni_once %f, ni_twice %f, ni_other %f, lmda %f, res_prob %f' % 
        #                (self.id2w[widl], self.id2w[widr], xij, xi_, delta, probability_discounted, ni_once, ni_twice, ni_other, lmda, res) )
        #    self.err_file.write(err_line + '\n')
        #    print err_line
        return res

    def __probability_word(self, wid):
        n_j = 0
        if wid in self.ug2n_r:
            n_j = self.ug2n_r[wid]
        res = float(n_j) / self.bg_num
        #assert res <= 1.0 # debug
        return res

    def __word2id(self, word, word_dict):
        if word in word_dict:
            return word_dict[word]
        else:
            return word_dict[self.unk_word]


