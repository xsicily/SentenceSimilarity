#-------------------------- define similarity measurement for python-----------------------#
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import numpy as np
import itertools
from itertools import chain
import text_clean

# define jaccard similarity for python #
def WebJsim(query, jdoc):
    intersection = set(query).intersection(set(jdoc))
    union = set(query).union(set(jdoc))
    return len(intersection)/len(union)


#-------include all the hyponyms and hypernyms of the associated words by quering the WordNet database--------#
# we use stopwds_filtered -list of word lists that is defined above
# as seed for wordnet

def get_expandwds(word_list):
    expand_list = []
    for ss in word_list: 
        hyper_list = []
        hypo_list = []
        new_list = []      
        mySynSets = wn.synsets(ss) #'ss.strip()'--->remove whitespace in the string
        for i, j in enumerate(mySynSets):    
        #print(i,j.name())
            hyper_list.extend(list(chain(*[i.lemma_names() for i in j.hypernyms()])))
        #print(hyper_list)
            hypo_list.extend(list(chain(*[i.lemma_names() for i in j.hyponyms()])))
        #print(hypo_list)
            new_list.extend(hypo_list)
            new_list.extend(hyper_list)
        expand_list.extend(new_list)
        expand_list.extend(word_list)
    return expand_list


