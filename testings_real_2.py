# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:02:03 2018

@author: hwpar5
"""

# importing function to list items in directory.
from os import listdir
from os.path import isfile, join
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from random import randrange

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import decomposition, ensemble

import pandas
import re
import math
import string
import textblob
import xgboost


def corpusMake():
    """
    This function creates a corpus using the files in a tirectory called text
    """
    # for each files in the directory, if they are from the texts folder, then dump it in the list
    # this function also auto sorts the list (hence why i had to make it text01, text02)
    files = [f for f in listdir('texts') if isfile(join('texts', f))]

    # then extract each files and we dump them in a list
    books = []
    for i in files:
        i = 'texts/' + i
        with open(i,'r') as f:
            doc = f.readlines()
        books.append(doc)
        doc = ''
    
    f = open('corpusdata.txt', 'a')
    #wipe everything before adding it to the list
    #or just create a new file everytime
    f.seek(0)
    f.truncate()
    # then we need to put everything into a file and turn it into a corpus
    for i in range(len(books)):
        book = str(books[i][0])
        f.write(str(book))
        f.write('\n')
    f.close()
    
    # a function to load the data
def loadData():
    """
    This function will load the corpus and puts it into a list
    """
    with open('corpusdata.txt','r') as f:
        doc = f.readlines()
    return doc
    
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def dataset_prep():
    data = open('corpusdata.txt').read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        content = line.split()
        texts.append(content)
    
    target_data = []
    for i in range(5):
        for j in range(10):
            target_data.append(i)
    
    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['label'] = target_data
    texts.pop()
    trainDF['text'] = texts
    trainDF['text']=[" ".join(review) for review in trainDF['text'].values]
    return trainDF

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def split_section(data, n):
    # split the dataset into training and validation datasets 
    if n == 0:
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data['text'], data['label'])
        # label encode the target variable 
        from sklearn import preprocessing
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
    if n == 1:
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data['text'], data['char_count'])
        # label encode the target variable 
        from sklearn import preprocessing
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
    if n == 2:
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data['text'], data['word_count'])
        # label encode the target variable 
        from sklearn import preprocessing
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
    if n == 3:
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data['text'], data['word_density'])
        # label encode the target variable 
        from sklearn import preprocessing
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
    if n == 4:
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data['text'], data['punctuation_count'])
        # label encode the target variable 
        from sklearn import preprocessing
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
    if n == 5:
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data['text'], data['title_word_count'])
        # label encode the target variable 
        from sklearn import preprocessing
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
    if n == 6:
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data['text'], data['upper_case_word_count'])
        # label encode the target variable 
        from sklearn import preprocessing
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
        
    return train_x, valid_x, train_y, valid_y

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def count_vector(train_x, valid_x, train_y, valid_y):
    count_vect = CountVectorizer(analyzer='word', lowercase=False, token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['text'])
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
    
    accuracy1 = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
    
    accuracy2 = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
    
    accuracy3 = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
    
    accuracy4 = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
    
    return accuracy, accuracy1, accuracy2, accuracy3, accuracy4

    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

def tf_idf_vectors(train_x, valid_x, train_y, valid_y):
    accuracy, accuracy1, accuracy2, accuracy3, accuracy4 = tf_idf_word(train_x, valid_x, train_y, valid_y)
    accuracy5, accuracy6, accuracy7, accuracy8, accuracy9 = tf_idf_ngram(train_x, valid_x, train_y, valid_y)
    #tf_idf_char(train_x, valid_x, train_y, valid_y)
    return accuracy, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9

def tf_idf_word(train_x, valid_x, train_y, valid_y):
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
    #print("NB, WordLevel TF-IDF: ", accuracy)
    accuracy1 = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
    #print("LR, WordLevel TF-IDF: ", accuracy1)
    accuracy2 = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
    #print("SVM, WordLevel TF-IDF: ", accuracy2)
    accuracy3 = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
    #print("RF, WordLevel TF-IDF: ", accuracy3)
    accuracy4 = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
    #print("Xgb, WordLevel TF-IDF: ", accuracy4)
    return accuracy, accuracy1, accuracy2, accuracy3, accuracy4
    
def tf_idf_ngram(train_x, valid_x, train_y, valid_y):
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    #print("NB, N-Gram Vectors: ", accuracy)
    accuracy1 = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    #print("LR, N-Gram Vectors: ", accuracy)
    accuracy2 = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    #print("SVM, N-Gram Vectors: ", accuracy)
    accuracy3 = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    #print("RF, N-Gram Vectors: ", accuracy3)
    accuracy4 = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram.tocsc(), train_y, xvalid_tfidf_ngram.tocsc())
    #print("Xgb, WordLevel TF-IDF: ", accuracy4)
    return accuracy, accuracy1, accuracy2, accuracy3, accuracy4
    
def tf_idf_char(train_x, valid_x, train_y, valid_y):
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF['text'])
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)
    
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print("NB, CharLevel Vectors: ", accuracy)
    
    accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print("LR, CharLevel Vectors: ", accuracy)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def nlp_features(trainDF):
    trainDF['char_count'] = trainDF['text'].apply(len)
    trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
    trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
    trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
    trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
    #trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
    #trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
    #trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
    #trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
    #trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))
    return trainDF
    
def check_pos_tag(x, flag):
    pos_family = {
        'noun' : ['NN','NNS','NNP','NNPS'],
        'pron' : ['PRP','PRP$','WP','WP$'],
        'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
        'adj' :  ['JJ','JJR','JJS'],
        'adv' : ['RB','RBR','RBS','WRB']
    }

    cnt = 0
    
    wiki = textblob.TextBlob(x)
    for tup in wiki.tags:
        ppo = list(tup)[1]
            
        if ppo in pos_family[flag]:
            cnt += 1

    return cnt
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def output_file(name, accuracy, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14):    
    output = open(name,'w')
    output.seek(0)
    output.truncate()

    output.write("NB, Count Vectors: ")
    output.write(str(accuracy))
    output.write("\n")
    output.write("LR, Count Vectors: ")
    output.write(str(accuracy1))
    output.write("\n")
    output.write("SVM, Count Vectors: ")
    output.write(str(accuracy2))
    output.write("\n")
    output.write("RF, Count Vectors: ")
    output.write(str(accuracy3))
    output.write("\n")
    output.write("XGB, Count Vectors: ")
    output.write(str(accuracy4))
    output.write("\n")
    output.write("\n")
    output.write("NB, WordLevel TF-IDF: ")
    output.write(str(accuracy5))
    output.write("\n")
    output.write("LR, WordLevel TF-IDF: ")
    output.write(str(accuracy6))
    output.write("\n")
    output.write("SVM, WordLevel TF-IDF: ")
    output.write(str(accuracy7))
    output.write("\n")
    output.write("RF, WordLevel TF-IDF: ")
    output.write(str(accuracy8))
    output.write("\n")
    output.write("XGB, WordLevel TF-IDF: ")
    output.write(str(accuracy9))
    output.write("\n")
    output.write("\n")
    output.write("NB, N-Gram Vectors: ")
    output.write(str(accuracy10))
    output.write("\n")
    output.write("LR, N-Gram Vectors: ")
    output.write(str(accuracy11))
    output.write("\n")
    output.write("SVM, N-Gram Vectors: ")
    output.write(str(accuracy12))
    output.write("\n")
    output.write("RF, N-Gram Vectors: ")
    output.write(str(accuracy13))
    output.write("\n")
    output.write("XGB, N-Gram Vectors: ")
    output.write(str(accuracy14))
    output.close()
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    
    corpusMake()
    corpus = loadData()
    
    trainDF = dataset_prep()
    trainDF = nlp_features(trainDF)
    train_x, valid_x, train_y, valid_y = split_section(trainDF, 0)
    accuracy, accuracy1, accuracy2, accuracy3, accuracy4 = count_vector(train_x, valid_x, train_y, valid_y)
    accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14 = tf_idf_vectors(train_x, valid_x, train_y, valid_y)    
    output_file('test_results/5author_result1.txt', accuracy, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14)

    train_x, valid_x, train_y, valid_y = split_section(trainDF, 1)
    accuracy, accuracy1, accuracy2, accuracy3, accuracy4 = count_vector(train_x, valid_x, train_y, valid_y)
    accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14 = tf_idf_vectors(train_x, valid_x, train_y, valid_y)    
    output_file('test_results/5author_result2.txt', accuracy, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14)
    
    train_x, valid_x, train_y, valid_y = split_section(trainDF, 2)
    accuracy, accuracy1, accuracy2, accuracy3, accuracy4 = count_vector(train_x, valid_x, train_y, valid_y)
    accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14 = tf_idf_vectors(train_x, valid_x, train_y, valid_y)    
    output_file('test_results/5author_result3.txt', accuracy, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14)
    
    train_x, valid_x, train_y, valid_y = split_section(trainDF, 3)
    accuracy, accuracy1, accuracy2, accuracy3, accuracy4 = count_vector(train_x, valid_x, train_y, valid_y)
    accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14 = tf_idf_vectors(train_x, valid_x, train_y, valid_y)    
    output_file('test_results/5author_result4.txt', accuracy, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14)
    
    train_x, valid_x, train_y, valid_y = split_section(trainDF, 4)
    accuracy, accuracy1, accuracy2, accuracy3, accuracy4 = count_vector(train_x, valid_x, train_y, valid_y)
    accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14 = tf_idf_vectors(train_x, valid_x, train_y, valid_y)    
    output_file('test_results/5author_result5.txt', accuracy, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14)
    
    train_x, valid_x, train_y, valid_y = split_section(trainDF, 5)
    accuracy, accuracy1, accuracy2, accuracy3, accuracy4 = count_vector(train_x, valid_x, train_y, valid_y)
    accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14 = tf_idf_vectors(train_x, valid_x, train_y, valid_y)    
    output_file('test_results/5author_result6.txt', accuracy, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14)
    
    train_x, valid_x, train_y, valid_y = split_section(trainDF, 6)
    accuracy, accuracy1, accuracy2, accuracy3, accuracy4 = count_vector(train_x, valid_x, train_y, valid_y)
    accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14 = tf_idf_vectors(train_x, valid_x, train_y, valid_y)    
    output_file('test_results/5author_result7.txt', accuracy, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, accuracy10, accuracy11, accuracy12, accuracy13, accuracy14)