# Semantic Annotation of ACM research papers using ACM classification tree

**Semantic annotation** is done through first representing words and documents in the vector space model using word2vec and doc2vec implementations, the vectors are taken as features into a classifier, trained and a model is made which can classify a document with acm classification tree 2012 categories.

## Setup Instructions
```
    $ workon myvirtualenv                                  [Optional]
	$ pip3 install -r requirements.txt
```

## Building the Model
```
    $ python3 run.py
```

## Classifying the Model
```
    $ python3 classify.py
```

## Researh Paper
Quoc V. Le, and Tomas Mikolov, ''Distributed Representations of Sentences and Documents ICML", 2014

Link :- https://cs.stanford.edu/~quocle/paragraph_vector.pdf
