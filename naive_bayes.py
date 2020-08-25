# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 14:33:14 2020

@author: Brar
"""

from model import clean_post,filter_token,get_tokenized_dataset,read_word_to_phoneme,get_dataset
import numpy as np
import subprocess
import os
import time
from nltk import word_tokenize
import random

tokenized_dataset=get_tokenized_dataset()
word_to_phoneme_dict=read_word_to_phoneme()
phoneme_vocabulary=set([word_to_phoneme_dict.values()])


def generate_phoneme_class_dict(tokenized_dataset,word_to_phoneme_dict):
    phoneme_class_count_dict={}
    for post,label in tokenized_dataset:
        for word in post:
            if word not in word_to_phoneme_dict:
                continue
            key=(word_to_phoneme_dict[word],label)
            phoneme_class_count_dict[key]=phoneme_class_count_dict.get(key,0)+1
    return phoneme_class_count_dict

phoneme_class_count=generate_phoneme_class_dict(tokenized_dataset,word_to_phoneme_dict)

def positive_negative_count(tokenized_dataset):
    YES='Yes'
    NO='No'
    pos_count=len([label for (post,label) in tokenized_dataset if label==YES])
    neg_count=len([label for (post,label) in tokenized_dataset if label==NO])
    return (pos_count,neg_count)


def get_phoneme_for_word(word):
    os.chdir('C:\Program Files (x86)\eSpeak\command_line')
    with open('phoneme.txt', 'w') as file:
        process = subprocess.Popen(['espeak.exe', '-xq',word], stdout=file)
    file.close()
    time.sleep(0.1)
    with open('phoneme.txt','r+') as file:
        phoneme=file.read()
        phoneme=phoneme.strip()
    file.close()
    return phoneme

def get_vocab_count(phoneme_class_count):
    YES='Yes'
    NO='No'
    pos_vocab_count=sum([count for (_,label),count in phoneme_class_count.items() if label==YES])
    neg_vocab_count=sum([count for (_,label),count in phoneme_class_count.items() if label==NO])
    return (pos_vocab_count,neg_vocab_count)
        

pos_count,neg_count=positive_negative_count(tokenized_dataset)


log_prior=np.log(pos_count/len(tokenized_dataset))-np.log(neg_count/len(tokenized_dataset))

def get_prob_for_word(word,phoneme_class_count,vocabulary_count,pos_vocab_count,neg_vocab_count):
    phoneme=get_phoneme_for_word(word)
    pos_prob=(phoneme_class_count.get((phoneme,'Yes'),0)+1)/(pos_vocab_count+vocabulary_count)
    neg_prob=(phoneme_class_count.get((phoneme,'No'),0)+1)/(neg_vocab_count+vocabulary_count)
    return np.log(pos_prob)-np.log(neg_prob)
    
def predict_post(post,phoneme_class_count,vocabulary_count,log_prior):
    pos_vocab_count,neg_vocab_count=get_vocab_count(phoneme_class_count)
    
    post=clean_post(post)
    post=word_tokenize(post)
    post=[filter_token(token) for token in post]
    
    log_probability=log_prior
    for word in post:
        log_probability+=get_prob_for_word(word,phoneme_class_count,vocabulary_count,pos_vocab_count,
                                                  neg_vocab_count)
    return log_probability

def get_class(log_probability):
    if log_probability>0 :
        return 'Yes' 
    return 'No'

def get_validation_dataset(tokenized_dataset):
    random.shuffle(tokenized_dataset)
    test_dataset=tokenized_dataset[:500]
    return test_dataset

def get_accuracy(test_data,phoneme_class_count,vocabulary_count,log_prior):
    correct_count=0
    for post,true_label in test_data:
        label=get_class(predict_post(post,phoneme_class_count,vocabulary_count,log_prior))
        if label==true_label:
            correct_count+=1
    return correct_count/len(test_data)


dataset=get_dataset()
print(get_accuracy(get_validation_dataset(dataset),phoneme_class_count,len(phoneme_vocabulary)
        ,log_prior))

        


    



        
