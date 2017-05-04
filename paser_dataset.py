#!/usr/bin/python
from __future__ import print_function
import json
from pprint import pprint
import sys
import numpy as np
from gensim import models
import re
from nltk.tokenize import TweetTokenizer

def clean_review(string):
    string = re.sub(r'[-!$%^&*()_+|~=`{}\[\]:\";<>?,.\/]', " SYMBOL ", string)
    string = re.sub(r'\d+.?\d*', " NUMBER ", string)

    return string.strip()


def make_clean_reviews(json_file):
    with open('clean.json', 'w') as fout:
        with open(json_file) as fin:
            for line in fin:
                d = json.loads(line)
                index = d[u'stars']
                review = d[u'text']
                review = clean_review(review)

                d[u'text'] = review
                tknzr = TweetTokenizer()
                tknzr_rv = tknzr.tokenize(review)
                if len(tknzr_rv) > 50 and index != 4:
                    out = json.dumps(d)
                    fout.write(out+"\n")

def word_to_vec(json_file, w):
    json_review_matrix_name = json_file.replace(".json","_matrix")
    json_review_labels_name = json_file.replace(".json","_label")

    with open(json_file) as fin:
        sentences_list = []

        labels_list=[]
        for line in fin:
            d = json.loads(line)
            index = d[u'stars']

            if index < 4:
                labels_list.append([1,0])
            else:
                labels_list.append([0,1])


            review = d[u'text']
            tknzr = TweetTokenizer()
            tknzr_rv = tknzr.tokenize(review)

            sentence_list = []

            for tk_index in xrange(50):
                try:
                    tk = tknzr_rv[tk_index]
                    vec = w[tk]
                except:
                    vec = np.random.rand(300)
                sentence_list.append(vec)
                
            sentences_list.append(sentence_list)


        sentences_matrix = np.asarray(sentences_list)
        np.save(json_review_matrix_name, sentences_matrix)    
        labels_matrix=np.asarray(labels_list).astype("int32")
        np.save(json_review_labels_name, labels_matrix)

def broadcast_to_3_dimension(matrix_name):
    matrix_new_name = matrix_name.replace(".npy", "_3_dim")
    matrix = np.load(matrix_name, mmap_mode="r")
    matrix = np.transpose(matrix, (0, 2, 1))
    np.save(matrix_new_name, matrix)

def main():
    json_file=sys.argv[1]
    #w = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #make_clean_reviews(json_file)
    #word_to_vec(json_file, w)
    broadcast_to_3_dimension(json_file)

if __name__=="__main__":
    main()
