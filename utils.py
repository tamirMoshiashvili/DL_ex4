import json

import numpy as np

UNK = 'UNK'  # sign of UNK in glove


def read_glove_file(filename):
    """
    :param filename: name of glove-file
    :return: dict, word (string) to numpy-vector.
    """
    w2v = dict()
    with open(filename, 'r') as f:
        for line in f:
            line = line.split()
            word = line.pop(0)
            w2v[word] = np.array(map(float, line))
    return w2v


def sentence_to_matrix(s, w2v):
    """
    :param s: sentence, each word separated by space.
    :param w2v: word-to-vector dict.
    :return: matrix representing the sentence, each ith-row is a vector for ith-word.
    """
    s = s.split()
    return np.array([w2v[w] if w in w2v
                     else w2v[UNK]]
                    for w in s)


def read_snli_data(file_path, w2v):
    """
    :param file_path: file to read from.
    :param w2v: word-to-vector dict.
    :return: list of tuples, each is (sentence1, sentence2, label), where sentence-i is a matrix.
    """
    data = []

    with open(file_path, 'r') as f:
        for line in f:
            line = json.load(line)
            label = line['gold_label']
            if label != u'-':  # ignore lines with no gold-label
                s1 = sentence_to_matrix(line['sentence1'], w2v)
                s2 = sentence_to_matrix(line['sentence2'], w2v)

                data.append((s1, s2, label))
    return data
