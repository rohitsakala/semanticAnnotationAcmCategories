#!/usr/bin/env python
import urllib, json
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from random import shuffle
from gensim.models import Phrases
from nltk.corpus import stopwords
import nltk
import string

bigram = Phrases()
cat_list = ['computer_vision', 'artificial_intelligence', 'information_extraction', 'databases', 'computer_science']

base_url = 'https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro=&explaintext=&titles='

categories_list = ['scientific_computing', 'human-computer_interaction', \
    'real_time_and_embedded_systems', 'algorithms_and_theory', \
    'networks_and_communications', 'simulation', 'computer_education', \
    'machine_learning_and_pattern_recognition', \
    'natural_language_and_speech', 'artificial_intelligence', \
    'information_retrieval', 'world_wide_web', 'multimedia', \
    'distributed_and_parallel_computing', 'security_and_privacy', \
    'data_mining', 'databases', 'programming_languages', \
    'hardware_and_architecture', 'software_engineering', \
    'operating_systems', 'computer_vision', 'graphics', \
    'bioinformatics_and_computational_biology']

proxies = {}
proxies['http'] = "http://proxy.iiit.ac.in:8080"
proxies['https'] = "http://proxy.iiit.ac.in:8080"

summary_category = {}

mapping_dict = {}
mapping_dict['scientific_computing']='Computational_science'
mapping_dict['real_time_and_embedded_systems']="embedded_system"
mapping_dict['distributed_and_parallel_computing'] = "Distributed_computing"
mapping_dict['algorithms_and_theory'] = "algorithm"
mapping_dict['networks_and_communications'] = "computer_network"
mapping_dict['machine_learning_and_pattern_recognition'] = "Pattern_recognition"
mapping_dict['natural_language_and_speech'] = "Natural_language_processing"
mapping_dict['security_and_privacy'] = "Computer_security"
mapping_dict['hardware_and_architecture'] = "Hardware_architecture"
mapping_dict['bioinformatics_and_computational_biology'] = "Computational_biology"

def get_summary(category):
    global proxies, base_url, mapping_dict, bigram

    if category in mapping_dict:
        category=mapping_dict[category]

    url = base_url + str(category)
    response = urllib.urlopen(url, proxies=proxies)

    data = json.loads(response.read())

    try:
        query_data = data['query']['pages']

        page_id = query_data.keys()[0]

        sentence = [ word.encode('utf-8', 'ignore').decode('utf-8') 
            for word in nltk.word_tokenize(query_data[page_id]['extract'].lower()) 
            if word not in string.punctuation]
        return sentence

    except Exception, e:
        print query_data
        print page_id
        print category
        print data
        raise e

for category in cat_list:
    summary_category[category] = get_summary(category)

for category in categories_list:
    summary_category[category] = get_summary(category)

class LabeledLineSentence(object):

    def __init__(self, summary_category):
        self.summary_category = summary_category

    def to_array(self):
        self.sentences = []

        for category in cat_list:
            self.sentences.append(LabeledSentence(self.summary_category[category], \
                [category]))

        for category in categories_list:
            self.sentences.append(LabeledSentence(self.summary_category[category], \
                [category]))

        return self.sentences


    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


sentences = LabeledLineSentence(summary_category)

model = Doc2Vec(dm=0, min_count=1, \
    window=10, size=100, sample=1e-4, negative=5, workers=7)

model.build_vocab(sentences.to_array())

for epoch in range(100):
    model.train(sentences.sentences_perm())

final_dict = {}

vocab = list(model.vocab.keys())
print vocab
print model.most_similar(positive='computer_vision'.split('_'))

for category in categories_list:
    if category in mapping_dict:
        category=mapping_dict[category]

    val = []
    for main_cat in cat_list:
        val.append([model.similarity(category, \
            main_cat), main_cat])

    final_dict[category] = max(val, key=lambda x:x[0])[1]


print final_dict