'''
    This code is written to model the two architectures
    mentioned in the research paper (Refer Readme)
    on the corpus.
'''

# loading gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# loading numpy
#import numpy

# loading other modules
from random import shuffle
import logging
import os.path
import sys

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
    filename="log_out.txt", filemode="a+")
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

# Class to modify the corpus to give it as an input to doc2vec
class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

    def __iter__(self):
        for source in sources:
            with utils.smart_open(source) as fin:
                title = ''
                category = ''
                field = ''
                for line in fin.readlines():
                    line = line.decode("utf-8")
                    if line == '\r\n':
                        yield LabeledSentence(\
                            utils.to_unicode(title).split(),
                            [category, field])
                        title = ''
                        category = ''
                        field = ''

                    if line.startswith('#*'):
                        title = str(line).strip()[2:]

                    if line.startswith('#c'):
                        category = str(line).strip()[2:]

                    if line.startswith('#f'):
                        field = str(line).strip()[2:]

    def to_array(self):
        self.sentences = []

        for source in sources:
            with utils.smart_open(source) as fin:
                title = ''
                category = ''
                field = ''
                for line in fin.readlines():
                    line = line.decode("utf-8")
                    if line == '\r\n':
                        self.sentences.append(LabeledSentence(
                            utils.to_unicode(title).split(), [category, field]))
                        title = ''
                        category = ''
                        field = ''

                    if line.startswith('#*'):
                        title = str(line).strip()[2:]

                    if line.startswith('#c'):
                        category = str(line).strip()[2:]

                    if line.startswith('#f'):
                        field = str(line).strip()[2:]

        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

# List of divided corpus
sources = ['dataset.txt']

# Converting paragraphs into required format
sentences = LabeledLineSentence(sources)

# creating model objects
model_skip = Doc2Vec(dm=0, min_count=1, \
    window=10, size=100, sample=1e-4, negative=5, workers=7)
model_cow = Doc2Vec(dm=1, min_count=1, window=10,
    size=100, sample=1e-4, negative=5, workers=7)

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
