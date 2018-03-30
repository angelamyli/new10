#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)

def get_train_data(vocabulary, batch_size, num_steps):
    ########################## This is the start line of my code ############################################

    batch_num = len(vocabulary) // batch_size
    sample_len = batch_size * batch_num

    # we set the next word as the label
    # It doesn't matter what value it gets, most likely it will be discarded in the following steps.
    x_array = np.array(vocabulary[:sample_len])
    x_array = x_array.reshape([batch_size, -1])

    if sample_len == len(vocabulary) :
        vocabulary.extend(vocabulary[0])

    y_array = np.array(vocabulary[1:sample_len+1])
    y_array = y_array.reshape([batch_size, -1])

    epoch_size = batch_size // num_steps
    for i in range(epoch_size):
        x = x_array[:, num_steps*i : num_steps*(i+1)]
        y = y_array[:, num_steps*i : num_steps*(i+1)]
        yield(x, y)

    ########################## This is the bottom line of my code ############################################



def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
