#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    Handle Preprocessing for the
    ACM Dataset(dataset.txt)

'''

# loading gensim modules
from gensim import utils
from gensim.parsing.porter import PorterStemmer

import string
import pickle
import re


stopwords_list = pickle.load(open('stopwords.p', 'rb'))

def remove_stopwords(sentence):
    '''
        Stopword Removal
    '''
    sentence = utils.to_unicode(sentence)
    return " ".join(w for w in sentence.split() if w not in stopwords_list)

RE_PUNCT = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)
def strip_punctuation(sentence):
    '''
        Remove Puntuations
    '''
    sentence = utils.to_unicode(sentence)
    return RE_PUNCT.sub(" ", sentence)


RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
def strip_tags(sentence):
    '''
        Remove Tags
    '''
    sentence = utils.to_unicode(sentence)
    return RE_TAGS.sub("", sentence)

def strip_short(sentence, minsize=3):
    '''
        Split and Join words
        if size >= minsize
    '''
    sentence = utils.to_unicode(sentence)
    return " ".join(e for e in sentence.split() if len(e) >= minsize)


RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
def strip_numeric(sentence):
    '''
        Strip Numerical Values
    '''
    sentence = utils.to_unicode(sentence)
    return RE_NUMERIC.sub("", sentence)

RE_NONALPHA = re.compile(r"\W", re.UNICODE)
def strip_non_alphanum(sentence):
    '''
        Remove Non-Alphanumeric Characters.
    '''
    sentence = utils.to_unicode(sentence)
    return RE_NONALPHA.sub(" ", sentence)

RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)
def strip_multiple_whitespaces(sentence):
    '''
        Remove Multiple Whitespaces
    '''
    sentence = utils.to_unicode(sentence)
    return RE_WHITESPACE.sub(" ", sentence)

RE_AL_NUM = re.compile(r"([a-z]+)([0-9]+)", flags=re.UNICODE)
RE_NUM_AL = re.compile(r"([0-9]+)([a-z]+)", flags=re.UNICODE)

def split_alphanum(sentence):
    sentence = utils.to_unicode(sentence)
    sentence = RE_AL_NUM.sub(r"\1 \2", sentence)
    return RE_NUM_AL.sub(r"\1 \2", sentence)

def stem_text(text):
    """
    Return lowercase and (porter-)stemmed version of string `text`.
    """
    text = utils.to_unicode(text)
    stemmer = PorterStemmer()
    return ' '.join(stemmer.stem(word) for word in text.split())

DEFAULT_FILTERS = [lambda x: x.lower(), strip_tags, \
            strip_punctuation, strip_multiple_whitespaces,
            strip_numeric, remove_stopwords, strip_short, stem_text]


def preprocess_string(sentence, filters=DEFAULT_FILTERS):
    '''
        Preprocessing String by running all the default filters.
    '''
    sentence = utils.to_unicode(sentence)
    for f in filters:
        sentence = f(sentence)
    return sentence.split()
