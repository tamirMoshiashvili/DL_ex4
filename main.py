from time import time

import sys

import SnliModel
import utils
import dynet as dy
import numpy as np

if __name__ == '__main__':
    """ args:
        path_to_train path_to_dev path_to_test
        data/snli_1.0_train.txt data/snli_1.0_dev.txt data/snli_1.0_test.txt
    """

    t = time()
    print 'start'

    args = sys.argv[1:]
    train = utils.read_from(args[0])
    dev = utils.read_from(args[1])
    test = utils.read_from(args[2])

    print 'time to read files:', time() - t
    t = time()

    word_set = utils.read_vocab_file('vocab.txt')
    w2i = {w: i for i, w in enumerate(word_set)}

    labels = ['neutral', 'entailment', 'contradiction']
    l2i = {l: i for i, l in enumerate(labels)}

    vecs = np.loadtxt('wordVectors.txt')

    pc = dy.ParameterCollection()
    model = SnliModel.SnliModel(pc, w2i, l2i, pre_embed=vecs)
    model.train_on(train, dev, test, to_save=True, model_name='model')

    print 'time to run model:', time() - t
