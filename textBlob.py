#!/usr/bin/python
from __future__ import print_function
import json
from pprint import pprint
import sys
from textblob import TextBlob


def main():
    json_file=sys.argv[1]

    labels=[]
    reviews=[]
    reviewPolarity=[]
    with open(json_file) as fin:
        for line in fin:
            d = json.loads(line)
            index = d[u'stars']
            review = d[u'text']
            labels.append(index)
            reviews.append(review)
            blob = TextBlob(review)
            polaritySum = 0
            for sentence in blob.sentences:
                #print(sentence.sentiment.polarity)
                polaritySum = polaritySum + sentence.sentiment.polarity
                reviewPolarity.append(polaritySum)
            #print(blob.tags)
            #print(blob.noun_phrases)

    with open('polarity.txt','w') as fout:
        for i in xrange(len(labels)):
       	    new_string = str(labels[i])+"\t"+str(reviewPolarity[i])
	    fout.write(new_string)
	    print(new_string)

#import pickle

#with open('outfile', 'wb') as fp:
#    pickle.dump(itemlist, fp)

# To read the above file
#with open ('outfile', 'rb') as fp:
#    itemlist = pickle.load(fp)

#polFile = open('polarity.txt', 'w')
#def printSentencePolarity(blob):
#    for sentence in blob.sentences:
#        print(sentence.sentiment.polarity)
#        print(sentence.sentiment)


if __name__=="__main__":
    main()


