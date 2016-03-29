################ This code is written to model the two architectures mentioned in the research paper(Refer Readme) on the corpus #######################

# loading gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# loading numpy
import numpy

# loading other modules
from random import shuffle
import logging
import os.path
import sys

# Class to modify the corpus to give it as an input to doc2vec
class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources
        flipped = {}
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

# List of divided corpus
sources = {'testNegative.txt':'TEST_NEG', 'testPositive.txt':'TEST_POS', 'trainNegative.txt':'TRAIN_NEG', 'trainPositive.txt':'TRAIN_POS', 'trainUnsup.txt':'TRAIN_UNS'}

# Converting paragraphs into required format
sentences = LabeledLineSentence(sources)

# creating model objects
model_skip = Doc2Vec(dm=0,min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
model_cow = Doc2Vec(dm=1,min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

# Building word vectors with initial random weights
model_cow.build_vocab(sentences.to_array())
model_skip.build_vocab(sentences.to_array())

# Training the models
for epoch in range(50):
    logger.info('Epoch %d' % epoch)
    model_cow.train(sentences.sentences_perm())
    model_skip.train(sentences.sentences_perm())

# Saving models for future use
model_skip.save('./imdb_skip.d2v')
model_cow.save('./imdb_cow.d2v')
