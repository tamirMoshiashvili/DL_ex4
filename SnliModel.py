from time import time

import dynet as dy
import numpy as np
import pickle

UNK = '_UNK_'


class SnliModel(object):
    def __init__(self, pc, w2i, l2i, emb_dim=50,
                 f_in_dim=25, f_act=dy.rectify, g_out_dim=15):
        self.model = pc
        self.w2i = w2i
        self.w2i[UNK] = len(w2i)
        self.l2i = l2i
        self.i2l = {i: l for l, i in l2i.iteritems()}

        self.spec = (emb_dim, f_in_dim, f_act, g_out_dim)

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
        return (dy.transpose(w1) * w2).npvalue()[0]

    @staticmethod
    def beta_i(ea_i, b_vecs):
        """
        :param ea_i: vector as list
        :param b_vecs: list of vectors.
        """
        soft_ea_i = dy.softmax(dy.inputTensor(ea_i))
        return dy.esum([soft_eij * b_j for soft_eij, b_j in zip(soft_ea_i.npvalue(), b_vecs)])

    @staticmethod
    def alpha_j(eb_j, a_vecs):
        """
        :param eb_j: vector as list
        :param a_vecs: list of vectors.
        """
        soft_eb_j = dy.softmax(dy.inputTensor(eb_j))
        return dy.esum([soft_eij * a_i for soft_eij, a_i in zip(soft_eb_j.npvalue(), a_vecs)])

    def feed_forward_g(self, u, v):
        p_W, p_b = dy.parameter(self.g_W), dy.parameter(self.g_b)
        return p_W * dy.concatenate([u, v]) + p_b

    @staticmethod
    def get_v(list_of_vis):
        """ for calculate v1 and v2 """
        return dy.esum(list_of_vis)

    def feed_forward_h(self, v1, v2):
        p_W, p_b = dy.parameter(self.h_W), dy.parameter(self.h_b)
        return p_W * dy.concatenate([v1, v2]) + p_b

    @staticmethod
    def find_index_label(net_output):
        return np.argmax(net_output.npvalue())

    def apply_embed(self, s):
        """ s is a list of words """
        return [dy.lookup(self.embed, self.w2i[w]) if w in self.w2i
                else dy.lookup(self.embed, self.w2i[UNK])
                for w in s]

    def __call__(self, a, b):
        """ a and b are sentences """
        a = a.split()
        b = b.split()

        # embed
        a_vecs = self.apply_embed(a)
        b_vecs = self.apply_embed(b)

        # F
        a_after_f = [self.feed_forward_f(w_vec) for w_vec in a_vecs]
        b_after_f = [self.feed_forward_f(w_vec) for w_vec in b_vecs]

        # e
        ea, eb = dict(), dict()
        for a_f_i in a_after_f:
            ea[a_f_i] = [self.e_i_j(a_f_i, b_f_j) for b_f_j in b_after_f]
        for j, b_f_j in enumerate(b_after_f):
            eb[b_f_j] = [ea[a_f_i][j] for a_f_i in ea]

        # alpha and beta
        beta_list = [self.beta_i(ea[ea_i], b_vecs) for ea_i in ea]
        alpha_list = [self.alpha_j(eb[eb_j], a_vecs) for eb_j in eb]

        # G
        v1_vecs = [self.feed_forward_g(a_i, beta_i) for a_i, beta_i in zip(a_vecs, beta_list)]
        v2_vecs = [self.feed_forward_g(b_j, alpha_j) for b_j, alpha_j in zip(b_vecs, alpha_list)]

        # aggregate - v1 and v2
        v1 = self.get_v(v1_vecs)
        v2 = self.get_v(v2_vecs)

        # H
        probs = self.feed_forward_h(v1, v2)
        return probs

    def train_on(self, train, dev, test, to_save=False, model_name=None):
        """ train, dev and test are tuples (s1 sentences, s2 sentences, gold labels)
            if to_save is True, then user must specify model_name
        """
        if to_save and model_name is None:
            raise Exception('to save, you must specify model_name')

        report_dev_file = open('acc_dev_and_test', 'w')
        report_dev_file.write('dev,test\n')
        best_dev_acc = 0.0
        report_train_file = open('acc_train', 'w')
        report_train_file.write('train\n')

        trainer = dy.AdagradTrainer(self.model)

        for epoch in range(1):
            train_size = 0
            total_loss = good = bad = 0.0
            t = time()

            for s1, s2, gold_label in train:
                if gold_label == '-':  # ignore input that has no gold-label
                    continue

                dy.renew_cg()
                train_size += 1

                output = self(s1, s2)
                pred_label = self.find_index_label(output)
                loss = -dy.log(dy.pick(pred_label, self.l2i[gold_label]))
                total_loss += loss.value()
                loss.backward()
                trainer.update()

                if gold_label == self.i2l[pred_label]:
                    good += 1
                else:
                    bad += 1

                if train_size % 500 == 499:
                    curr_dev_acc, test_acc = self.check_test(dev, 'dev'), self.check_test(test, 'test')
                    report_dev_file.write(str(curr_dev_acc) + ',' + str(test_acc) + '\n')
                    if to_save and curr_dev_acc > best_dev_acc:
                        best_dev_acc = curr_dev_acc
                        self.save_model(model_name)

            train_acc = good / (good + bad)
            print epoch, 'loss:', total_loss / train_size, 'acc:', train_acc, 'time:', time() - t
            report_train_file.write(str(train_acc) + '\n')

        report_dev_file.close()
        report_train_file.close()

    def check_test(self, test, name):
        """ test is a tuple (s1 sentences, s2 sentences, gold labels) """
        good = bad = 0.0
        t = time()
        test_size = 0

        for s1, s2, gold_label in test:
            if gold_label == '-':
                continue

            dy.renew_cg()
            test_size += 1

            output = self(s1, s2)
            pred_label = self.i2l[self.find_index_label(output)]
            if gold_label == self.i2l[pred_label]:
                good += 1
            else:
                bad += 1

        acc = good / (good + bad)
        print 'acc on ' + name + ':', acc, 'time:', time() - t
        return acc

    def save_model(self, filename):
        self.model.save(filename)
        emb_dim, f_in_dim, f_act, g_out_dim = self.spec
        obj = {'w2i': self.w2i, 'l2i': self.l2i,
               'emb_dim': emb_dim, 'f_in_dim': f_in_dim, 'f_act': f_act, 'g_out_dim': g_out_dim}
        pickle.dump(obj, open(filename + '.params', 'wb'))

    @staticmethod
    def load_model(filename):
        reader = pickle.load(open(filename + '.params', 'rb'))
        w2i, l2i = reader['w2i'], reader['l2i']
        emb_dim, f_in_dim, f_act, g_out_dim = reader['emb_dim'], reader['f_in_dim'], reader['f_act'], reader[
            'g_out_dim']
        m = dy.ParameterCollection()

        net = SnliModel(m, w2i, l2i, emb_dim, f_in_dim, f_act, g_out_dim)
        m.populate(filename)

        return net


if __name__ == '__main__':
    t0 = time()
    print 'start'

    my_words = ['hello', 'see', 'me', 'look', 'right', 'here']
    w_to_i = {my_word: index for index, my_word in enumerate(my_words)}
    my_labels = ['a', 'b', 'c']
    l_to_i = {my_label: index for index, my_label in enumerate(my_labels)}

    my_snli_model = SnliModel(dy.ParameterCollection(), w_to_i, l_to_i)
    a1 = 'hello see me'.split()
    b1 = 'look right here'.split()
    dy.renew_cg()
    prob = my_snli_model(a1, b1)
    print my_snli_model.find_index_label(prob), prob.npvalue()

    print 'time to run all:', time() - t0
