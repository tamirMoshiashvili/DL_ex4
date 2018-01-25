import sys
from time import time

import dynet as dy

import SnliModel
import utils

LOAD_FLAG = '--load'
SAVE_FLAG = '--save'
TRAIN_FLAG = '--train'
TEST_FLAG = '--test'

if __name__ == '__main__':
    """
    args:
    glove_file [options]
    - glove_file - path to the glove file (glove.840B.300d.txt)

    options:
    * --save model_name
        add the flag '--save' to save the model in case of training,
        model_name is the name of the saved-model.
    * --load pretrained_model_name
        add the flag '--load' to load a model from a file,
        pretrained_model_name is the path to the file to load the model from.
    * --train train_file dev_file
        add the flag '--train' in order to train the model,
        train_file is the path to the train file,
        dev_file is the path to the dev file.
    * --test test_file
        add the flag '--test' in order to test the model on the file,
        test_file is the path to the test-file
    """

    args = sys.argv[1:]

    t = time()
    print 'start reading glove-file'
    glove_file = args[0]
    w2v = utils.read_glove_file(glove_file)
    print 'time to read Glove-file:', time() - t
    t = time()

    labels = ['neutral', 'entailment', 'contradiction']
    l2i = {l: i for i, l in enumerate(labels)}
    pc = dy.ParameterCollection()
    model = SnliModel.SnliModel(pc, l2i)

    # if load pre-trained model from file
    if LOAD_FLAG in args:
        pretrained_model_file = args[args.index(LOAD_FLAG) + 1]
        model.load(pretrained_model_file)

    # if to save the model into a file after training
    model_name = None
    if SAVE_FLAG in args:
        model_name = args[args.index(SAVE_FLAG) + 1]

    # if to train the model
    if TRAIN_FLAG in args:
        t = time()
        print '\nstart read snli-data'
        train_file = args[args.index(TRAIN_FLAG) + 1]
        dev_file = args[args.index(TRAIN_FLAG) + 2]
        train = utils.read_snli_data(train_file, w2v)
        dev = utils.read_snli_data(dev_file, w2v)
        print 'time to read snli files:', time() - t
        t = time()

        print '\nstart training'
        model.train_on(train, dev, model_name=model_name)

    # if to run a test
    if TEST_FLAG in args:
        t = time()
        print '\nstart read snli-data'
        test_file = args[args.index(TEST_FLAG) + 1]
        test = utils.read_snli_data(test_file, w2v)
        print 'time to read snli files:', time() - t
        t = time()

        acc, _ = model.check_test(test)
        print 'test accuracy:', acc

    print 'time to run model:', time() - t
