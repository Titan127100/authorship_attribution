# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 01:34:50 2018

@author: hwpar5
"""
from os import listdir
from os.path import isfile, join
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from random import randrange
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn import decomposition, ensemble
import xgboost




def corpusMake(n):
    """
    This function creates a corpus using the files in a tirectory called text
    """
    # for each files in the directory, if they are from the texts folder, then dump it in the list
    # this function also auto sorts the list (hence why i had to make it text01, text02)
    files = [f for f in listdir('texts-10') if isfile(join('texts-10', f))]

    # then extract each files and we dump them in a list
    books = []
    for i in files:
        if n == 10:
            i = 'texts-10/' + i
        else:
            i = 'texts-5/' + i
        with open(i,'r', encoding='utf8', errors='ignore') as f:
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

def count_vector(data, target_data):
    """
    This function will count vectorise the list
    """
    count_vect = CountVectorizer(analyzer='word', lowercase=False, token_pattern=r'\w{1,}')
    count_vect.fit(data)
    xtrain_count =  count_vect.transform(data)
    xtrain_count = xtrain_count.toarray()
    temp = []
    for i in range(len(xtrain_count)):
        temp.append(xtrain_count[i])
    return temp
    
def cross_validation_split(dataset_X, dataset_Y, k):
    """
    This is a cross validation function where it takes, as input a dataset and the number of folds it will divide the data into
    k folds.
    it returns a new list, where the data is divided into k folds
    UPDATE: We want to make it so that each fold will have at least one text from each author. Meaning that we will definitely have 10 folds with one author in each one
    """
    split_data_X = [] # shall store the split data here
    split_data_Y = [] # shall store the split author here
    temp_X = []
    temp_X = dataset_X # make a copy and add it to the a temporary list for the data part
    temp_Y = []
    temp_Y = dataset_Y # make a copy and add it to a temporary list for the author part
    fold_size = int(len(dataset_X) / k) # how many items in each group there should be
    for i in range(k):
        fold_X = [] # this list will store all the fold datas
        fold_Y = [] # this will store all the other data
        while len(fold_X) < fold_size:
            index = randrange(len(temp_X)) # generate a random number within the range of the length of the temporary data holder
            fold_X.append(temp_X.pop(index))
            fold_Y.append(temp_Y.pop(index))
            
        split_data_X.append(fold_X)
        split_data_Y.append(fold_Y)
    return split_data_X, split_data_Y

def loadData():
    """
    This function will load the corpus and puts it into a list
    """
    labels = []
    with open('corpusdata.txt','r') as f:
        doc = f.readlines()
        
    with open('authors/authors.txt', 'r') as f:
        auth = f.readlines()
    for i in range(len(auth)):
        labels.append(auth[i].split("\n"))    
    
    target_data = []
    for i in range(len(auth)):
        target_data.append(labels[i][0])
    
    return doc, target_data

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


def run_machine_learning(X_data, Y_data, k):
    """
    We divide up the dataset into 9 testings and 1 trainings. After we identify that we will use KNeighbours to do stuff
    """
    # so we also want to try using 13 testings and 37 training data.
    test_score = 0
    test_score1 = 0
    test_score2 = 0
    test_score3 = 0
    test_score4 = 0
    test_score5 = 0
    test_score6 = 0
    test_score7 = 0
    test_score8 = 0
    test_score9 = 0
    for i in range(k):
        temp_X = []
        temp_Y = []
        
        # make a temporary copy of the original dataset
        for j in range(len(X_data)):
            temp_X.append(X_data[j])
            temp_Y.append(Y_data[j])
        train_X, test_X, train_Y, test_Y = train_test_data(temp_X, temp_Y, i, k)
        
        test_score += train_model(KNeighborsClassifier(n_neighbors=3), train_X, train_Y, test_X, test_Y)
        
        test_score1 += train_model(tree.DecisionTreeClassifier(), train_X, train_Y, test_X, test_Y)
        
        test_score2 += train_model(RandomForestClassifier(), train_X, train_Y, test_X, test_Y)
        
        test_score3 += train_model(AdaBoostClassifier(), train_X, train_Y, test_X, test_Y)
        
        test_score4 += train_model(GaussianNB(), train_X, train_Y, test_X, test_Y)
        
        test_score5 += train_model(naive_bayes.MultinomialNB(), train_X, train_Y, test_X, test_Y)
        
        test_score6 += train_model(linear_model.LogisticRegression(), train_X, train_Y, test_X, test_Y)
    
        test_score7 += train_model(svm.SVC(), train_X, train_Y, test_X, test_Y)

        
    test_score /= 10
    test_score1 /= 10
    test_score2 /= 10
    test_score3 /= 10
    test_score4 /= 10
    test_score5 /= 10
    test_score6 /= 10
    test_score7 /= 10
    test_score8 /= 10
    test_score9 /= 10
    
    return test_score, test_score1, test_score2, test_score3, test_score4, test_score5, test_score6, test_score7, test_score8, test_score9

def train_test_data(X_data, Y_data, test_pos, k):
    """
    Here we split up the training data and testing data and make sure they are in the same list
    """
    # first, we want to create the training data
    # idea is that we want to grab the first quarter of the data from X_data and Y_data and we append each item to the
    # test list
    temp_X = []
    temp_Y = []
    # a list for the training
    test_data_X = []
    test_data_Y = []
    
    # quarter of the data floored to get the required position
    test_size = int(len(X_data) / 4)
    # start an iteration that grabs each k up to the test_size
    for j in range(test_size): # the temp_pos will keep a storage of the testing set via their respective position
        if j+test_pos >= len(X_data): # if it goes outside the index range then go back to the start.
            pos = j+test_pos-len(X_data)
        else:
            pos = j+test_pos
        temp_X.append(X_data[pos]) # then store it
        temp_Y.append(Y_data[pos])
        
    
    for k in range(len(temp_X)):
        for l in range(len(temp_X[k])):
            test_data_X.append(temp_X[k][l])
            test_data_Y.append(temp_Y[k][l])
            
    train_data_X = []
    train_data_Y = []
    for i in range(len(X_data)):
        for j in range(len(X_data[i])):
            train_data_X.append(X_data[i][j])
            train_data_Y.append(Y_data[i][j])
                
    return train_data_X, test_data_X, train_data_Y, test_data_Y

def output_file(name, accuracy, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9):  
    """
    We output to a file
    """
    output = open(name,'w')
    output.seek(0)
    output.truncate()

    output.write("KN, Count Vectors: ")
    output.write(str(accuracy))
    output.write("\n")
    output.write("DTC, Count Vectors: ")
    output.write(str(accuracy1))
    output.write("\n")
    output.write("RF, Count Vectors: ")
    output.write(str(accuracy2))
    output.write("\n")
    output.write("ABC, Count Vectors: ")
    output.write(str(accuracy3))
    output.write("\n")
    output.write("GNB, Count Vectors: ")
    output.write(str(accuracy4))
    output.write("\n")
    output.write("MNB, Count Vectors: ")
    output.write(str(accuracy5))
    output.write("\n")
    output.write("LR, Count Vectors: ")
    output.write(str(accuracy6))
    output.write("\n")
    output.write("SVM, Count Vectors: ")
    output.write(str(accuracy7))
    output.write("\n")

    output.close()

if __name__ == "__main__":
    corpusMake(10)
    data, target_data = loadData()
    """ choose which feature to use """
    xtrain_count = count_vector(data, target_data)
    #tfidf_transformer = TfidfTransformer()
    #X_train_tfidf = tfidf_transformer.fit_transform(xtrain_count)
    X_data, Y_data = cross_validation_split(xtrain_count, target_data, 10)
    #X_data, Y_data = cross_validation_split(X_train_tfidf, target_data, 10)
    
    test_score, test_score1, test_score2, test_score3, test_score4, test_score5, test_score6, test_score7, test_score8, test_score9 = run_machine_learning(X_data, Y_data, 10)
    output_file('test_results/10author_result.txt',test_score, test_score1, test_score2, test_score3, test_score4, test_score5, test_score6, test_score7, test_score8, test_score9)
    