# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:53:38 2020

@author: Brar
"""

from preprocess import clean_post,filter_token,get_tokenized_dataset,read_word_to_phoneme,get_dataset
import numpy as np
from nltk import word_tokenize
import random
from nltk.stem import PorterStemmer
import pandas as pd
import re
import spacy

POSITIVE_TRAIN_COUNT = 1950
NEGATIVE_TRAIN_COUNT = 6000
LOG_PRIOR = np.log(POSITIVE_TRAIN_COUNT/NEGATIVE_TRAIN_COUNT)

negative_data = pd.read_csv('Dataset/negative_posts.csv', error_bad_lines = False, encoding = "ISO-8859-1")

positive_data = pd.read_csv('Dataset/positive_posts.csv', error_bad_lines = False, encoding = "ISO-8859-1")

def get_posts_list (data) :
    posts_list = []
    for index in range(len(data)):
        posts_list.append(data.loc[index, "Posts"])
    return posts_list

def filter_posts (posts) :
    posts_list = []
    porter_stemmer = PorterStemmer()
    sp = spacy.load('en_core_web_sm')
    stopwords_set = sp.Defaults.stop_words
    for post in posts:
        post = clean_post(post)
        words = [porter_stemmer.stem(word) for word in word_tokenize(post) if word not in stopwords_set]
        posts_list.append(words)
    return posts_list

positive_posts_list = get_posts_list(positive_data)
negative_posts_list = get_posts_list(negative_data)

random.shuffle(negative_posts_list)
positive_posts_list = filter_posts(positive_posts_list)
negative_posts_list = filter_posts(negative_posts_list)

postive_train_posts = positive_posts_list[:POSITIVE_TRAIN_COUNT]
positive_test_posts = positive_posts_list[POSITIVE_TRAIN_COUNT:]

negative_train_posts = negative_posts_list[:NEGATIVE_TRAIN_COUNT]
negative_test_posts = negative_posts_list[NEGATIVE_TRAIN_COUNT:6500]

def generate_freq_dict (posts_list):
    freq_dict = {}
    for post in posts_list:
        for word in post:
            if (len(word) > 2):
                freq_dict[word] = freq_dict.get(word, 0) + 1
    return freq_dict

positive_dict = generate_freq_dict(positive_posts_list)
negative_dict = generate_freq_dict(negative_posts_list)

def get_word_count (freq_dict):
    count = 0
    for word, freq in freq_dict.items():
        count += freq
    return count

POSITIVE_WORD_COUNT = get_word_count(positive_dict)
NEGATIVE_WORD_COUNT = get_word_count(negative_dict)

def get_likelihood(word, freq_dict, word_count):
    vocab_len = len(freq_dict)
    return np.log((freq_dict.get(word, 1) + 1) / (word_count + vocab_len))

porter_stemmer = PorterStemmer()
sp = spacy.load('en_core_web_sm')
stopwords_set = sp.Defaults.stop_words
def predict_post (post, positive_dict, negative_dict):
    post = clean_post(post)
    words = [porter_stemmer.stem(word) for word in word_tokenize(post) if word not in stopwords_set and len(porter_stemmer.stem(word)) > 2]
    prob = LOG_PRIOR
    for word in words:
        prob += get_likelihood(word, positive_dict, POSITIVE_WORD_COUNT) - get_likelihood(word, negative_dict, NEGATIVE_WORD_COUNT)
    return prob > 0

def test_classifier (negative_test_posts, positive_test_posts, positive_dict, negative_dict):
    test_posts = []
    for post in negative_test_posts:
        test_posts.append((post, False))
    for post in positive_test_posts:
        test_posts.append((post, True))
    correct_count = 0
    for post, label in test_posts:
        post = ' '.join(post)
        predict_label = predict_post(post, positive_dict, negative_dict)
        if predict_label == label:
            correct_count += 1
    return correct_count / len(test_posts)

print(test_classifier(negative_test_posts, positive_test_posts, positive_dict, negative_dict))

print(predict_post('i wanna have love', positive_dict, negative_dict))