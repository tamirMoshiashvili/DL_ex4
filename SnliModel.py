from time import time

import dynet as dy
import numpy as np

UNK = 'UNK'


class SnliModel(object):
    def __init__(self, pc, l2i, emb_dim=300, hid_dim=200, act_func=dy.rectify):
        self.model = pc
        self.l2i = l2i
        self.i2l = {i: l for l, i in l2i.iteritems()}

        # embed to linear
        self.linear_embed_W, self.linear_embed_b = pc.add_parameters((hid_dim, emb_dim)), pc.add_parameters(hid_dim)

        # F
        self.f_W_in, self.f_b_in = pc.add_parameters((hid_dim, hid_dim)), pc.add_parameters(hid_dim)
        self.f_act = act_func
        self.f_W_out, self.f_b_out = pc.add_parameters((hid_dim, hid_dim)), pc.add_parameters(hid_dim)

        # G
        self.g_W_in, self.g_b_in = pc.add_parameters((hid_dim, 2 * hid_dim)), pc.add_parameters(hid_dim)
        self.g_act = act_func
        self.g_W_out, self.g_b_out = pc.add_parameters((hid_dim, hid_dim)), pc.add_parameters(hid_dim)

        # H
        self.h_W_in, self.h_b_in = pc.add_parameters((hid_dim, 2 * hid_dim)), pc.add_parameters(hid_dim)
        self.h_act = act_func
        self.h_W_out, self.h_b_out = pc.add_parameters((hid_dim, hid_dim)), pc.add_parameters(hid_dim)

        # to out-dim
        out_dim = len(l2i)
        self.linear_final_W, self.linear_final_b = pc.add_parameters((out_dim, hid_dim)), pc.add_parameters(out_dim)

    def apply_embed_linear(self, a, b):
        """
        :param a: matrix, each row is the vector representing the ith word in the sentence.
        :param b: matrix, each row is the vector representing the ith word in the sentence.
        :return: a, b after linear-layer, hid-dim
        """
        p_W_embed, p_b_embed = dy.parameter(self.linear_embed_W), dy.parameter(self.linear_embed_b)
        a = p_W_embed * dy.inputTensor(a) + p_b_embed
        b = p_W_embed * dy.inputTensor(b) + p_b_embed
        return a, b

    def apply_f(self, a, b, drop=0.2):
        """ MLP on each a, b """
        p_W_f_in, p_b_f_in = dy.parameter(self.f_W_in), dy.parameter(self.f_b_in)
        a_f_in = self.f_act(p_W_f_in * dy.dropout(a, drop) + p_b_f_in)
        b_f_in = self.f_act(p_W_f_in * dy.dropout(b, drop) + p_b_f_in)

        p_W_f_out, p_b_f_out = dy.parameter(self.f_W_out), dy.parameter(self.f_b_out)
        a_f_out = self.f_act(p_W_f_out * dy.dropout(a_f_in, drop) + p_b_f_out)
        b_f_out = self.f_act(p_W_f_out * dy.dropout(b_f_in, drop) + p_b_f_out)

        return a_f_out, b_f_out

    def apply_g(self, a, b, drop=0.2):
        """ MLP on each a, b """
        p_W_g_in, p_b_g_in = dy.parameter(self.g_W_in), dy.parameter(self.g_b_in)
        a_g_in = self.g_act(p_W_g_in * dy.dropout(a, drop) + p_b_g_in)
        b_g_in = self.g_act(p_W_g_in * dy.dropout(b, drop) + p_b_g_in)

        p_W_g_out, p_b_g_out = dy.parameter(self.g_W_out), dy.parameter(self.g_b_out)
        a_g_out = self.g_act(p_W_g_out * dy.dropout(a_g_in, drop) + p_b_g_out)
        b_g_out = self.g_act(p_W_g_out * dy.dropout(b_g_in, drop) + p_b_g_out)

        return a_g_out, b_g_out

    def apply_h(self, s, drop=0.2):
        """ MLP """
        p_W_h_in, p_b_h_in = dy.parameter(self.h_W_in), dy.parameter(self.h_b_in)
        h_in = self.h_act(p_W_h_in * dy.dropout(s, drop) + p_b_h_in)

        p_W_h_out, p_b_h_out = dy.parameter(self.h_W_out), dy.parameter(self.h_b_out)
        return self.h_act(p_W_h_out * dy.dropout(h_in, drop) + p_b_h_out)

    def __call__(self, a, b):
        """ a and b are each a matrix """
        # embed
        a, b = self.apply_embed_linear(a, b)

        # F
        a_f, b_f = self.apply_f(a, b)

        # attention
        a_atten_score = a_f * dy.transpose(b_f)
        a_atten = dy.softmax(a_atten_score)
        b_atten_score = dy.transpose(a_atten_score)
        b_atten = dy.softmax(b_atten_score)

        # align
        a_pairs = dy.concatenate_cols([a, a_atten * b])
        b_pairs = dy.concatenate_cols([b, b_atten * a])

        # G
        a_g, b_g = self.apply_g(a_pairs, b_pairs)

        # sum
        a_sum = dy.sum_dim(a_g, [0])
        b_sum = dy.sum_dim(b_g, [0])
        concat = dy.transpose(dy.concatenate([a_sum, b_sum]))

        # H
        sentence_h = self.apply_h(concat)

        # to out dim
        p_W_out, p_b_out = dy.parameter(self.linear_final_W), dy.parameter(self.linear_final_b)
        out = dy.transpose(p_W_out * sentence_h + p_b_out)
        return dy.softmax(out)

    def train_on(self, train, dev, epochs=1, model_name=None):
        """
        if model_name passed then the model will be saved.
        """
        report_dev_file = open('acc_dev.txt', 'w')
        report_dev_file.write('dev\n')
        best_dev_acc = 0.0
        report_train_file = open('acc_train.txt', 'w')
        report_train_file.write('train\n')

        trainer = dy.AdamTrainer(self.model)

        check_after = 60000
        train_size = len(train)

        for epoch in range(epochs):
            total_loss = good = 0.0
            t_epoch = t = time()

            for i, (s1, s2, gold_label) in enumerate(train):
                dy.renew_cg()

                output = self(s1, s2)
                loss = -dy.log(dy.pick(output, self.l2i[gold_label]))
                total_loss += loss.value()
                loss.backward()
                trainer.update()

                pred_label_index = np.argmax(output.npvalue())
                if gold_label == self.i2l[pred_label_index]:
                    good += 1

                if i % check_after == check_after - 1:
                    print 'time for', check_after, 'sentences:', time() - t
                    t = time()

                    # test_acc = self.check_test(test)
                    curr_dev_acc = self.check_test(dev)
                    report_dev_file.write(str(curr_dev_acc) + '\n')
                    if model_name and curr_dev_acc > best_dev_acc:
                        best_dev_acc = curr_dev_acc
                        self.save_model(model_name)
                    print 'time for dev:', time() - t, 'dev-acc:', curr_dev_acc
                    t = time()

            train_acc = good / train_size
            print epoch, 'loss:', total_loss / train_size, 'acc:', train_acc, 'time:', time() - t_epoch
            report_train_file.write(str(train_acc) + '\n')

            t = time()
            curr_dev_acc, test_acc = self.check_test(dev)
            report_dev_file.write(str(curr_dev_acc) + ',' + str(test_acc) + '\n')
            if model_name and curr_dev_acc > best_dev_acc:
                best_dev_acc = curr_dev_acc
                self.save_model(model_name)
            print 'time for dev and test:', time() - t, 'dev-acc:', curr_dev_acc, 'test-acc:', test_acc

        report_dev_file.close()
        report_train_file.close()

    def check_test(self, test):
        """ test is a tuple (s1 sentences, s2 sentences, gold labels) """
        good = 0.0
        test_size = len(test)

        for s1, s2, gold_label in test:
            dy.renew_cg()

            output = self(s1, s2)
            pred_label = self.i2l[np.argmax(output.npvalue())]
            if gold_label == pred_label:
                good += 1

        acc = good / test_size
        return acc

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model.populate(filename)
