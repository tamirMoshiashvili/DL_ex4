import sys
from time import time

import dynet as dy

import SnliModel
import utils

if __name__ == '__main__':
    """ args:
        path_to_train path_to_dev path_to_test
        data/snli_1.0_train.txt data/snli_1.0_dev.txt data/snli_1.0_test.txt
    """

    t = time()
    print 'start reading glove-file'

    glove_file = 'glove.840B.300d.txt'
    w2v = utils.read_glove_file(glove_file)

    print 'time to read Glove-file:', time() - t
    t = time()

    print '\nstart read snli-data'
    args = sys.argv[1:]
    train = utils.read_snli_data(args[0], w2v)
    dev = utils.read_snli_data(args[1], w2v)
    # test = utils.read_snli_data(args[2], w2i, vecs)

    labels = ['neutral', 'entailment', 'contradiction']
    l2i = {l: i for i, l in enumerate(labels)}

    print 'time to read input-files:', time() - t
    t = time()

    print '\nstart training'
    pc = dy.Model()
    model = SnliModel.SnliModel(pc, l2i)
    model.train_on(train, dev, model_name='model')

    print 'time to run model:', time() - t
