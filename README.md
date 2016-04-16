# Semantic Annotation of ACM research papers using ACM classification tree

**Semantic annotation** is done through first representing words and documents in the vector space model using word2vec and doc2vec implementations, the vectors are taken as features into a classifier, trained and a model is made which can classify a document with acm classification tree 2012 categories.

## Setup Instructions
```
    $ workon myvirtualenv                                  [Optional]
	$ pip3 install -r requirements.txt
```
Download the Dataset needed for ACM in the ACM Directory from [here](https://www.dropbox.com/s/91uc71wlhd4sg70/CS_Citation_Network.zip?dl=0).

## Building the Model
```
    $ python3 run.py
```

## Classifying the Model
```
    $ python3 classify.py
```

##Mentors:
- **Course Instructor:**
	- Vasudev Verma
- **TA:**
	- Priya Radhakrishnan

##Major Packages Required
- nltk
- gensim
- numpy
- scikit-learn
- pickle

**Members:**
- [Rohit SVK](https://github.com/rohitsakala)
- [Sharvil Katariya](https://github.com/scorpionhiccup)
- [Nikhil Chavanke](https://github.com/nikhilchavanke) 

## Research Paper
Quoc V. Le, and Tomas Mikolov, ''Distributed Representations of Sentences and Documents ICML", 2014

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean, “Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR”, 2013.

Cao et al., 2015, ''A Novel Neural Topic Model and Its Supervised Extension''. AAAI 2015

Link :- https://cs.stanford.edu/~quocle/paragraph_vector.pdf

Resources are available [here](resources.md).