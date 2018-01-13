from time import time

import sys

import SnliModel
import utils
import dynet as dy

if __name__ == '__main__':
    t = time()
    print 'start'

    args = sys.argv[1:]
    train = utils.read_from(args[0])
    dev = utils.read_from(args[1])
    test = utils.read_from(args[2])

    word_set1 = utils.word_set_from_sentences(train[0])
    word_set2 = utils.word_set_from_sentences(train[1])
    word_set1.update(word_set2)
    w2i = {w: i for i, w in enumerate(word_set1)}

    labels = ['neutral', 'entailment', 'contradiction']
    l2i = {l: i for i, l in enumerate(labels)}

    pc = dy.ParameterCollection()
    model = SnliModel.SnliModel(pc, w2i, l2i)

    print 'time to run all:', time() - t
