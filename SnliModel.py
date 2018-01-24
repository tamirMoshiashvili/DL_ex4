from time import time

import dynet as dy

UNK = 'UNK'


class SnliModel(object):
    def __init__(self, model, l2i, emb_dim=300, hid_dim=200):
        # embed_dim = w2v[UNK].shape[0] send this as emb dim
        self.model = model
        self.l2i = l2i
        self.i2l = {i: l for l, i in l2i.iteritems()}

        self.linear_embed = model.add_parameters((emb_dim, hid_dim))

        self.mlp_f = (model.add_parameters((hid_dim, hid_dim)), model.add_parameters((hid_dim, hid_dim)))
        self.mlp_g = (model.add_parameters((2 * hid_dim, hid_dim)), model.add_parameters((hid_dim, hid_dim)))
        self.mlp_h = (model.add_parameters((2 * hid_dim, hid_dim)), model.add_parameters((hid_dim, hid_dim)))
        self.act_func = dy.rectify

        out_dim = len(l2i)
        self.linear_final = model.add_parameters((hid_dim, out_dim))

    def apply_linear_embed(self, sent1, sent2):
        """
        :param sent1: np matrix.
        :param sent2: np matrix.
        :return: each sentence after projection to hid_dim.
        """
        p_W = dy.parameter(self.linear_embed)
        sent1 = dy.inputTensor(sent1) * p_W
        sent2 = dy.inputTensor(sent2) * p_W
        return sent1, sent2

    def apply_mlp_layer(self, sent1, sent2, layer_in, layer_out, drop=0.2):
        """
        :param sent1: np matrix.
        :param sent2: np matrix.
        :param layer_in: dy parameter (matrix).
        :param layer_out: dy parameter (matrix).
        :param drop: drop rate.
        :return: both sentence after going through mlp-layer.
        """
        sent1 = self.act_func(dy.dropout(sent1, drop) * layer_in)
        sent2 = self.act_func(dy.dropout(sent2, drop) * layer_in)

        sent1 = self.act_func(dy.dropout(sent1, drop) * layer_out)
        sent2 = self.act_func(dy.dropout(sent2, drop) * layer_out)
        return sent1, sent2

    def apply_f(self, sent1, sent2, drop=0.2):
        f_in, f_out = dy.parameter(self.mlp_f[0]), dy.parameter(self.mlp_f[1])
        return self.apply_mlp_layer(sent1, sent2, f_in, f_out, drop)

    def apply_g(self, sent1_combine, sent2_combine):
        g_in, g_out = dy.parameter(self.mlp_g[0]), dy.parameter(self.mlp_g[1])
        return self.apply_mlp_layer(sent1_combine, sent2_combine, g_in, g_out)

    def apply_h(self, input_combine, drop=0.2):
        h_in, h_out = dy.parameter(self.mlp_h[0]), dy.parameter(self.mlp_h[1])
        h = self.act_func(dy.dropout(input_combine, drop) * h_in)
        h = self.act_func(dy.dropout(h, drop) * h_out)
        return h

    def __call__(self, sent1, sent2):
        """
        :param sent1: np matrix.
        :param sent2: np matrix.
        :return: np array of 3 elements.
        """
        sent1_linear, sent2_linear = self.apply_linear_embed(sent1, sent2)
        f1, f2 = self.apply_f(sent1_linear, sent2_linear)

        score1 = f1 * dy.transpose(f2)
        prob1 = dy.softmax(score1)
        score2 = dy.transpose(score1)
        prob2 = dy.softmax(score2)

        sent1_combine = dy.concatenate_cols([sent1_linear, prob1 * sent2_linear])
        sent2_combine = dy.concatenate_cols([sent2_linear, prob2 * sent1_linear])

        # sum
        g1, g2 = self.apply_g(sent1_combine, sent2_combine)
        sent1_output = dy.sum_dim(g1, [0])
        sent2_output = dy.sum_dim(g2, [0])

        input_combine = dy.transpose(dy.concatenate([sent1_output, sent2_output]))
        h = self.apply_h(input_combine)

        linear_final = dy.parameter(self.linear_final)
        h = h * linear_final

        output = dy.log_softmax(dy.transpose(h))
        return output

    def train_on(self, train, dev, epochs=10, batch_size=32, model_name=None):
        """
        :param train: list of tuples (s1, s2, gold label),
                        s1 and s2 are each a matrix,
                        gold label is a string
        :param dev: same as train.
        :param epochs: number of epochs.
        :param batch_size: batch size.
        :param model_name: name of file if you want to save the model, otherwise None.
        """
        trainer = dy.AdamTrainer(self.model)
        best_dev_acc = self.check_on_dev(dev, model_name, 0.0)
        check_after = int(32000 / batch_size)
        curr_index = 0
        display_after = 10000
        train_size = len(train)

        for epoch in range(epochs):
            print 'start epoch:', epoch
            t_epoch = time()
            total_loss = good = 0.0
            dy.np.random.shuffle(train)

            dy.renew_cg()
            errors = []
            for i in range(train_size):
                if i % display_after == display_after - 1:
                    print 'current index:', i + 1

                if i % batch_size == batch_size - 1:
                    batch_error = dy.esum(errors)
                    total_loss += batch_error.value()
                    batch_error.backward()
                    trainer.update()

                    if curr_index == check_after:
                        curr_index = 0
                        best_dev_acc = self.check_on_dev(dev, model_name, best_dev_acc)
                    else:
                        curr_index += 1

                    dy.renew_cg()
                    errors = []

                s1, s2, gold_label = train[i]
                output = self(s1, s2)
                gold_label_index = self.l2i[gold_label]
                loss = dy.pickneglogsoftmax(output, gold_label_index)
                errors.append(loss)

                pred_label_index = dy.np.argmax(output.npvalue())
                if pred_label_index == gold_label_index:
                    good += 1

            if errors:
                batch_error = dy.esum(errors)
                total_loss += batch_error.value()
                batch_error.backward()
                trainer.update()

            best_dev_acc = self.check_on_dev(dev, model_name, best_dev_acc)
            print 'train - acc:', good / train_size,\
                'loss:', total_loss / train_size,\
                'time:', time() - t_epoch

    def check_on_dev(self, dev, model_name, best_dev_acc):
        print 'start checking on dev'
        curr_dev_acc, t_dev = self.check_test(dev)
        print 'dev - acc:', curr_dev_acc, 'time:', t_dev

        if model_name and curr_dev_acc > best_dev_acc:
            best_dev_acc = curr_dev_acc
            self.save(model_name)
        return best_dev_acc

    def check_test(self, test):
        """
        :param test: list of tuples (s1, s2, gold label),
                        s1 and s2 are each a matrix,
                        gold label is a string
        :return: accuracy on the test and time to run
        """
        t_test = time()
        good = 0.0
        test_size = len(test)

        for s1, s2, gold_label in test:
            dy.renew_cg()

            output = self(s1, s2)
            pred_label = self.i2l[dy.np.argmax(output.npvalue())]
            if gold_label == pred_label:
                good += 1

        acc = good / test_size
        return acc, time() - t_test

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)
