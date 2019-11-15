#------------------------------------------ Sentence Similarity----------------------------------------------#
##############################################################################################################
# This script is used to calculate sentence similarity of the original sentence pairs from the targeted dataset
# The similarity measurement methods include 
# 1-Jaccard similarity
# 2-Expanded Jaccard similarity involving hypernyms & hyponyms based on WordNet
# 3-Semantic similarity: 
# *Li, Y., McLean, D., Bandar, Z. A., O'shea, J. D., & Crockett, K. (2006). 
# *Sentence similarity based on semantic nets and corpus statistics. 
# 4-Wikipedia similarity

from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import csv
import text_clean
import Jacca_sim
import wiki_similarity as wikisim
from nltk.tokenize import word_tokenize
import Semantic_Similarity as seSim
import semantic_os as sim

# read csv files
def csv_2_list(filename):
    data = []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if any(row):    # remove the empty row
                data.append(list(row))
    return data

# get sentence pair (sp) as list
def get_data(data):
    sp = []
    s = []
    ratings = []
    for i in range(1, len(data)):
        s1 = data[i][1]
        s2 = data[i][2]
        s3 = data[i][3]
        s = [s1, s2]
        ratings.append(float(s3))
        sp.append(s)
    return sp, ratings

def new_text_clean(expand_list):
    new_text = text_clean.listAsString(expand_list)
    temp1 = text_clean.compound_split(new_text)
    temp2 = text_clean.tokenizer(temp1)
    #temp3 = text_clean.remove_stopwords(temp2)
    #temp4 = text_clean.stemmer(temp3)
    return temp2


#-------------------------------------define main() function-------------------------------------#
if __name__ == "__main__":
    # get sentence pairs and human ratings
    filename = "./dataset/stss_131.csv"
    data = csv_2_list(filename)
    sp = get_data(data)[0]
    ratings = get_data(data)[1]
    ratings = np.asarray(ratings)

    # METHOD 1 ---Jaccard similarity
    sentencejsim_list = []
    for i in range(0, len(sp)):
        s1 = sp[i][0]
        s2 = sp[i][1]
        s1 = text_clean.remove_punct(s1)
        s2 = text_clean.remove_punct(s2)
        s1 = text_clean.tokenizer(s1)
        s2 = text_clean.tokenizer(s2)
        s1 = text_clean.remove_stopwords(s1)
        s2 = text_clean.remove_stopwords(s2)
        s1 = text_clean.stemmer(s1)
        s2 = text_clean.stemmer(s2)
        sentencejsim = Jacca_sim.WebJsim(s1,s2)
        sentencejsim_list.append(sentencejsim)
    # calculate pearson coefficiency
    sentence_JaccardSim = np.asarray(sentencejsim_list)
    print(sentence_JaccardSim)
    r_sentencejsim = pearsonr(ratings,sentence_JaccardSim) 
    print("r_jsim" + str(r_sentencejsim))

    # METHOD 2 --- Expand Jaccard similarity
    processed_sp = []
    for i in range(0,len(sp)):
        temp = []
        for s in sp[i]:
            temp1 = text_clean.compound_split(s)
            temp2 = text_clean.remove_punct(temp1)
            temp3 = text_clean.remove_nonalpha(temp2)
            temp4 = text_clean.tokenizer(temp3)
            temp.append(temp4)
        processed_sp.append(temp)
    HHJSim_list = []
    for pair_no in range(len(processed_sp)):
        words_bag_1 = processed_sp[pair_no][0]
        words_bag_2 = processed_sp[pair_no][1]
        new_text_1 = new_text_clean(Jacca_sim.get_expandwds(words_bag_1))
        new_text_2 = new_text_clean(Jacca_sim.get_expandwds(words_bag_2))
        hhsim = Jacca_sim.WebJsim(new_text_1, new_text_2)
        #print(hhsim)
        HHJSim_list.append(hhsim)
    # calculate pearson coefficiency
    HHJSim = np.asarray(HHJSim_list)
    print(HHJSim)
    r_hhjsim = pearsonr(ratings,HHJSim)
    print("r_hhjsim" + str(r_hhjsim))

    # METHOD 3 --- Semantic similarity
    wnsim_list = []
    for i in range(0, len(sp)):
        s1 = sp[i][0]
        s2 = sp[i][1]
        wnsim = seSim.SemanticSim(s1,s2)
        #wnsim = sim.similarity(s1,s2)
        wnsim_list.append(wnsim)
    # calculate pearson coefficiency
    wnsim = np.asarray(wnsim_list)
    print(wnsim)
    r_wnsim = pearsonr(ratings,wnsim)
    print("r_wnsim" + str(r_wnsim))

    # METHOD 4 --- Wikipedia similarity
    WikiCosSim_list = []
    for i in range(0, len(sp)):
        s1 = sp[i][0]
        s2 = sp[i][1]
        wikicos_sim = wikisim.WikicosSim(s1,s2)
        WikiCosSim_list.append(wikicos_sim)
    # calculate pearson coefficiency
    wikisim = np.asarray(WikiCosSim_list)
    print(wikisim)
    r_wikisim = pearsonr(ratings,wikisim)
    print("r_wikisim" + str(r_wikisim))
