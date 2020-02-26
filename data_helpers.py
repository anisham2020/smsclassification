from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import math
import unittest
import os,csv,io, re
from nltk.corpus import stopwords, words, wordnet
from nltk.stem import WordNetLemmatizer
import random
from tempfile import gettempdir
from nltk.tokenize import word_tokenize
import zipfile
import numpy as np
from six.moves import urllib
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf
import sys
"""reload(sys)
sys.setdefaultencoding('utf8')
"""
lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))
vocabularySize = 0
y = []
x = []
words = dict()
count_words = dict()

label_map = {'BILL_PAYMENT':1, 'BOOKING':2, 'RECEIPTS':3, 'BANKING':4, 'PROMOTIONAL':5,'REMINDER':6, 'OTP':7, 'OTHER':8};
def convert_y(y):
    y_ = []
    for label in y:
        temp = np.zeros(len(label_map))
        if(label in label_map):
            index = label_map[label] - 1
            temp[index] = 1.0
            y_.append(temp)
    return y_

def keep_important(x):
    x_ = []
    for i in x:
        i = i.split(" ")
        st1 = ""
        for j in i:
            if(j):
                if(count_words[j] < 10):
                    j = "UNK"
                st1+=" " + j; 
        x_.append(st1)
    return x_

def remove_stopwords(data):
    splitSentence = word_tokenize(data)
    filteredSentence = []
    for word in splitSentence:
        if(word not in stopWords):
            filteredSentence.append(word)
    return ' '.join(filteredSentence)

def clean_input(string):
    """
    Tokenization,removing whitespaces,useless characters from dataset.
    """
    string = re.sub(r'http\S+', '', string)
    string = re.sub(r'https\S+', '', string)
    string = re.sub(r"[^A-Za-z]+", " ", string)
    return string.strip().lower()

def read_data(file):
    
    id = 1
    counttt = 0
    data = csv.reader(io.open(file, 'rt', encoding = "UTF-8"),
                        delimiter = ",", quotechar = '"')
    for line in data:
        # print(line)

        sms = remove_stopwords(clean_input(line[2]))
        # print(counttt)
        for word in word_tokenize(sms):
            if((word in words)==False):
                count_words[word] = 1
                words[word] = id
                id+=1  
            else:
                count_words[word]+=1
        smsSender = line[1]
        smsClass = line[0]
        if(smsClass in label_map):
            x.append(sms)
            y.append(smsClass)
    # print(words['make']) 
   
    return words,count_words

def get_data():
    return [keep_important(x), convert_y(y)]
    
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def build_dataset(words, n_words):

    count = [[UNK,-1]]  
    count = (collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    for word in words:
        index = dictionary.get(word, 0)
        data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


