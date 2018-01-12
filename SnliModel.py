from time import time

import dynet as dy
import numpy as np


class SnliModel(object):
    def __init__(self, pc, w2i, l2i, emb_dim=30,
                 f_in_dim=30, f_act=dy.rectify,
                 g_out_dim=30):
        self.model = pc
        self.w2i = w2i
        self.l2i = l2i
        self.i2l = {i: l for l, i in l2i.iteritems()}

        self.embed = pc.add_lookup_parameters((len(w2i), emb_dim))

        # F
        self.f_W_in, self.f_b_in = pc.add_parameters((f_in_dim, emb_dim)), pc.add_parameters(f_in_dim)
        self.f_act = f_act
        self.f_W_out, self.f_b_out = pc.add_parameters((emb_dim, f_in_dim)), pc.add_parameters(emb_dim)

        # G
        self.g_W, self.g_b = pc.add_parameters((g_out_dim, 2 * emb_dim)), pc.add_parameters(g_out_dim)

        # H
        out_dim = len(l2i)
        self.h_W, self.h_b = pc.add_parameters((out_dim, 2 * g_out_dim)), pc.add_parameters(out_dim)

    def feed_forward_f(self, word):
        p_W_in, b_in = dy.parameter(self.f_W_in), dy.parameter(self.f_b_in)
        p_W_out, b_out = dy.parameter(self.f_W_out), dy.parameter(self.f_b_out)
        return p_W_out * self.f_act(p_W_in * word + b_in) + b_out

    @staticmethod
    def e_i_j(w1, w2):
        return w1 * w2

    @staticmethod
    def beta_i(ea_i):
        """ ea_i is a dict that map a vector to a matrix (?)
            :return vector with size of embed dim
        """
        beta_i_vecs = [dy.softmax(ea_i[b_j]) * b_j for b_j in ea_i]
        return dy.esum(beta_i_vecs)

    @staticmethod
    def alpha_j(eb_j):
        """ eb_j is a dict that map a vector to a matrix (?)
            :return vector with size of embed dim
        """
        alpha_j_vecs = [dy.softmax(eb_j[a_i]) * a_i for a_i in eb_j]
        return dy.esum(alpha_j_vecs)

    def feed_forward_g(self, u, v):
        p_W, p_b = dy.parameter(self.g_W), dy.parameter(self.g_b)
        return p_W * dy.concatenate(u, v) + p_b

    @staticmethod
    def get_v(list_of_vis):
        """ for calculate v1 and v2 """
        return dy.esum(list_of_vis)

    def feed_forward_h(self, v1, v2):
        p_W, p_b = dy.parameter(self.h_W), dy.parameter(self.h_b)
        return p_W * dy.concatenate(v1, v2) + p_b

    @staticmethod
    def find_index_label(net_output):
        return np.argmax(net_output.npvalue())

    def __call__(self, a, b):
        """ a and b are list of words (each word is a string) """
        # embed
        a_vecs = [dy.lookup(self.embed, self.w2i[w]) for w in a]
        b_vecs = [dy.lookup(self.embed, self.w2i[w]) for w in b]

        # F
        a_after_f = [self.feed_forward_f(w_vec) for w_vec in a_vecs]
        b_after_f = [self.feed_forward_f(w_vec) for w_vec in b_vecs]

        # e
        ea, eb = dict(), dict()
        for a_vec in a_after_f:
            ea_a_vec = dict()
            for b_vec in b_after_f:
                if b_vec not in eb:
                    eb[b_vec] = dict()

                eij = self.e_i_j(a_vec, b_vec)
                ea_a_vec[b_vec] = eij
                eb[b_vec][a_vec] = eij
            ea[a_vec] = ea_a_vec

        # alpha and beta
        beta_list = [self.beta_i(ea_i) for ea_i in ea]
        alpha_list = [self.alpha_j(eb_j) for eb_j in eb]

        # G
        v1_vecs = [self.feed_forward_g(a_i, beta_i) for a_i, beta_i in zip(a_vecs, beta_list)]
        v2_vecs = [self.feed_forward_g(b_j, alpha_j) for b_j, alpha_j in zip(b_vecs, alpha_list)]

        # aggregate - v1 and v2
        v1 = self.get_v(v1_vecs)
        v2 = self.get_v(v2_vecs)

        # H
        probs = self.feed_forward_h(v1, v2)
        return probs


if __name__ == '__main__':
    t0 = time()
    print 'start'

    my_words = ['hello', 'see', 'me', 'look', 'right', 'here']
    w_to_i = {my_word: index for index, my_word in enumerate(my_words)}
    my_labels = ['a', 'b', 'c']
    l_to_i = {my_label: index for index, my_label in enumerate(my_labels)}

    my_snli_model = SnliModel(dy.ParameterCollection(), w_to_i, l_to_i)
    prob = my_snli_model('hello see me'.split())
    print my_snli_model.find_index_label(prob), prob

    print 'time to run all:', time() - t0
