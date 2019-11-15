#------------------- Text processing-------------------------#
#input: sentences
# Text processing
import nltk
import string
import math 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
import re
import string

def compound_split(text):
    # change compound words to separate words ie. 'conditional-statements' -> 'conditional', 'statements'
    regex = re.compile("[-_]")
    trimmed = regex.sub(' ', text)
    return trimmed

# remove punctuations
def remove_punct(text):
    exclude = set(string.punctuation)
    s_no_punct = ''.join(ch for ch in text if ch not in exclude)
    return s_no_punct

# remove non-alpha
def remove_nonalpha(text):
    regex = re.compile('[^a-zA-Z]')
    nonAlphaRemoved = regex.sub(' ', text)
    return nonAlphaRemoved

# tokenize
def tokenizer(sentence):
    return word_tokenize(sentence.lower())

# remove stopwords
def remove_stopwords(word_list):
    english_stop_words = set(stopwords.words('english'))
    processed_word_list = []
    for word in word_list:
        if word not in english_stop_words:
            processed_word_list.append(word)
    return processed_word_list

def stemmer(word_list):
    #PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in word_list]
    return stemmed

def listAsString(list_string):
    one_string = ''
    for idx, word in enumerate(list_string):
        if idx == len(list_string) - 1:
            one_string = one_string + word
        else:
            temp_string = word + ' '
            one_string = one_string + temp_string 
    return one_string

if __name__ == "__main__":
    text = "I like to eat one-half apple."
    result = tokenizer_customize(text)
    print(result)
