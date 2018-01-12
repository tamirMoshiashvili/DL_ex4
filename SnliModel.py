import dynet as dy
import numpy as np


class SnliModel(object):
    def __init__(self, pc, w2i, l2i, emb_dim=100,
                 f_in_dim=100, f_act=dy.rectify, f_out_dim=100,
                 g_out_dim=100):
        self.model = pc
        self.w2i = w2i
        self.l2i = l2i
        self.i2l = {i: l for l, i in l2i.iteritems()}

        self.embed = pc.add_lookup_parameters((len(w2i), emb_dim))

        # F
        self.f_W_in, self.f_b_in = pc.add_parameters((f_in_dim, emb_dim)), pc.add_parameters(f_in_dim)
        self.f_act = f_act
        self.f_W_out, self.f_b_out = pc.add_parameters((f_out_dim, f_in_dim)), pc.add_parameters(f_out_dim)

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
        return dy.dot_product(w1, w2)

    @staticmethod
    def beta_i(e_i, b_j):
        """ e_i is a vector [ea_i_1, ea_i_2, ..., ea_i_j, ... ea_i_lenb]
            :return vector with size of embed dim
        """
        norm = dy.softmax(e_i)
        return dy.dot_product(norm, b_j)

    @staticmethod
    def alpha_j(e_j, a_i):
        """ e_j is a vector [eb_j_1, eb_j_2, ..., eb_j_i, ... eb_j_lena]
            :return vector with size of embed dim
        """
        norm = dy.softmax(e_j)
        return dy.dot_product(norm, a_i)

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

                dot_prod = self.e_i_j(a_vec, b_vec)
                ea_a_vec[b_vec] = dot_prod
                eb[b_vec][a_vec] = dot_prod
            ea[a_vec] = ea_a_vec

        # TODO continue with alpha and beta
