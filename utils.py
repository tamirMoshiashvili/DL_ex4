from time import time

import pandas as pd
import sys


def read_from(file_path):
    """ read the sentences out of the file
        sentence can be accessed by sentences[i]
    """
    df = pd.read_csv(file_path, delimiter='\t', usecols=['gold_label', 'sentence1', 'sentence2'])
    return df.sentence1, df.sentence2, df.gold_label


def word_set_from_sentences(sentences):
    word_set = set()

    for sent in sentences:
        words = sent.split()
        word_set.update(words)

    return list(word_set)


if __name__ == '__main__':
    t0 = time()
    print 'start'

    filepath = sys.argv[1]
    s1, s2, labels = read_from(filepath)
    ws = word_set_from_sentences(s1)
    print len(ws)
    print s1[:5]
    for wi in ws[:10]:
        print wi

    print 'time to run all:', time() - t0
