'''
    This code is written to model the two architectures
    mentioned in the research paper (Refer Readme)
    on the corpus.
'''

# loading gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# loading other modules
from random import shuffle
import logging
import os.path
import sys
import pickle

from acm_preprocess import preprocess_string

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
    filename="log_output.txt", filemode="w+")
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


mapping_dict = {'scientific_computing': 'OT', 'information_retrieval': 'IR', 'world_wide_web': 'IR', 'security_and_privacy': 'OT', 'computer_education': 'AI', 'distributed_and_parallel_computing': 'OT', 'programming_languages': 'OT', 'algorithms_and_theory': 'AI', 'networks_and_communications': 'OT', 'machine_learning_and_pattern_recognition': 'AI', 'multimedia': 'IR', 'natural_language_and_speech': 'AI', 'artificial_intelligence': 'AI', 'real_time_and_embedded_systems': 'OT', 'bioinformatics_and_computational_biology': 'CV', 'hardware_and_architecture': 'OT', 'data_mining': 'DB', 'graphics': 'CV', 'software_engineering': 'OT', 'computer_vision': 'CV', 'human-computer_interaction': 'OT', 'simulation': 'OT', 'databases': 'DB', 'operating_systems': 'OT'}

# Class to modify the corpus to give it as an input to doc2vec
class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources
        self.labels = []

    def __iter__(self):
        for source in sources:
            with utils.smart_open(source) as fin:
                title = ''
                conference = ''
                field = ''
                abstract = ''
                for line in fin.readlines():
                    line = line.decode("utf-8")
                    if line == '\r\n':
                        yield LabeledSentence(\
                            preprocess_string(title), \
                            tags=[conference, field])
                        title = ''
                        conference = ''
                        field = ''

                    if line.startswith('#*'):
                        title = str(line).strip()[2:]

                    if line.startswith('#c'):
                        conference = str(line).strip()[2:]

                    if line.startswith('#f'):
                        field = str(line).strip()[2:]

    def to_array(self):
        self.sentences = []
        self.labels = []
        
        global mapping_dict

        for source, prefix in sources.items():
            count = 0
            with utils.smart_open(source) as fin:
                title = ''
                conference = ''
                field = ''
                abstract = ''

                for line in fin.readlines():
                    line = line.decode("utf-8")
                    if line == '\r\n':
                        if abstract:
                            #field_preprocessed = preprocess_string(field)

                            self.labels.append(mapping_dict[field])

                            self.sentences.append(LabeledSentence(
                                preprocess_string(abstract), 
                                ["{0}_{1}".format(prefix, str(count)), mapping_dict[field]]))

                            count = count + 1
                            abstract=''

                        title = ''
                        conference = ''
                        field = ''
                        continue

                    if line.startswith('#*'):
                        title = str(line).strip()[2:]

                    if line.startswith('#c'):
                        conference = str(line).strip()[2:]

                    if line.startswith('#f'):
                        field = str(line).strip()[2:]

                    if line.startswith('#!'):
                        abstract = str(line).strip()[2:]
            fin.close()

        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

def storeData(outfile_path, saved_data):
    with open(outfile_path, 'wb') as outfile:
        pickle.dump(saved_data, \
            outfile, protocol=pickle.HIGHEST_PROTOCOL)

# Dictionary of divided corpus
sources = {'CS_Citation_Network': 'SET'}

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

storeData('labels.p', sentences.labels)

print(len(sentences.sentences), len(sentences.labels))

# Training the models
for epoch in range(10):
    logger.info('Epoch %d' % (epoch + 1))
    model_cow.train(sentences.sentences_perm())
    model_skip.train(sentences.sentences_perm())

# Saving models for future use
model_skip.save('./acm_skip.d2v')
model_cow.save('./acm_cow.d2v')