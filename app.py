# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 21:23:12 2018

@author: hwpar5
"""

from flask import Flask, render_template, flash, redirect, url_for, session, logging, request
from data import Articles
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from werkzeug import secure_filename
from nltk.corpus import stopwords
from sklearn import model_selection, linear_model
from sklearn import preprocessing
from os import listdir
from os.path import isfile, join
from sklearn.feature_extraction.text import CountVectorizer

import pandas
import os
import re
from xgboost.sklearn import XGBClassifier

app = Flask(__name__)

Articles = Articles()

@app.route('/')
def index():
    return render_template('index.html')

class RegisterForm(Form):
    author = StringField('Name', [validators.Length(min=1, max=50)])
    
@app.route('/results', methods=['GET', 'POST'])
def results():
    return render_template('results.html')

@app.route('/learning', methods=['GET', 'POST'])
def learning():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        author = form.author.data
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(filename))
        author = learnBooks(filename, "english", author)
        return redirect(url_for('results'))
    return render_template('learning.html', form=form)

@app.route('/testing', methods=['GET', 'POST'])
def testing():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(filename))
        author = startFeatureExtraction(filename, "english")
        return render_template('testing.html', author = author[0])
    return render_template('testing.html')

@app.route('/spanishtesting', methods=['GET', 'POST'])
def spanishtesting():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(filename))
        author = startFeatureExtraction(filename, "spanish")
        return render_template('testing_spanish.html', author = author[0])
    return render_template('testing_spanish.html')

@app.route('/spanishlearning', methods=['GET', 'POST'])
def spanishlearning():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        author = form.author.data
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(filename))
        author = learnBooks(filename, "spanish", author)
        return redirect(url_for('results'))
    return render_template('learning_spanish.html', form=form)

def learnBooks(file, language, author):
    """
    The learn Books function will accept a file, language and the author as input and it will store it in the correct file locations.
    """
    corpusMake(language)
    texts, authors = loadData(language)
    inputText = loadText(file)
    texts, authors = manageInfo(texts, authors, inputText, author)
    updateInfo(texts, authors, language, inputText)
    
def updateInfo(texts, authors, language, inputText):
    """
    This folder will update the authors and text into its respetive folders.
    """
    filename = ""
    if language == "spanish":
        if len(texts) < 10:
            filename = "tests-spanish/text0" + str(len(texts)) + ".txt"
        else:
            filename = "tests-spanish/text" + str(len(texts)) + ".txt"
    elif language == "english":
        if len(texts) < 10:
            filename = "texts-10/text0" + str(len(texts)) + ".txt"
        else:
            filename = "texts-10/text" + str(len(texts)) + ".txt"
    output = open(filename,'w', encoding='utf8', errors='ignore')
    output.seek(0)
    output.truncate()
    output.write(str(inputText[0]))
    output.close()

    filename = ""
    if language == "spanish":
        if len(texts) < 10:
            filename = "authors-spanish/authors-spanish.txt"
        else:
            filename = "authors-spanish/authors-spanish.txt"
    elif language == "english":
        if len(texts) < 10:
            filename = "authors/authors.txt"
        else:
            filename = "authors/authors.txt"
    output = open(filename,'w', encoding='utf8', errors='ignore')
    output.seek(0)
    output.truncate()
    for i in range(len(authors)):
        output.write("\n")
        output.write(str(authors[i]))
    output.close()

def manageInfo(texts, authors, inputText, author):
    """
    Manages the files so that they will be added to the directory correctly.
    """
    texts.append(inputText[0])
    authors.append(author)
    return texts, authors

def loadText(file):
    """
    We need to get the contents of the input file so that we can put it in the directory.
    returns the contents of the file.
    """
    with open("%s" % file, 'r', encoding='utf8', errors='ignore') as f:
        file_content = f.readlines()
        
    return file_content

def loadData(language):
    """
    Grab all texts in the directory currently for later usage
    returns the corpus and authors
    """
    with open('corpusdata.txt','r') as f:
        doc = f.readlines()
        
    if language == "english":
        with open('authors/authors.txt', 'r', encoding='utf8', errors='ignore') as f:
            auth = f.readlines()
    elif language == "spanish":
        with open('authors-spanish/authors-spanish.txt', 'r', encoding='utf8', errors='ignore') as f:
            auth = f.readlines()
    
    return doc, auth

#-------------------------------------------------------------------------------------------------------------------------------------------

def startFeatureExtraction(file, language):
    """
    Using machine learning techniques, this function will perform other functions on user input
    returns the predicted author to the user.
    """
    corpusMake(language)
    trainDF, UInput, train_y= dataset_prep(file, language)
    train_x = trainDF['text']
    valid_x = UInput['text']
    author = count_vector(train_x, valid_x, train_y, trainDF)
    if len(author)==0:
        author.append("We cannot find the author")
    return author
    #predictions = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
    #print("Xgb, Count Vectors: ", accuracy4)

def count_vector(train_x, valid_x, train_y, trainDF):
    """
    This function will count vectorise the texts and they will apply logistic regression on the the input data.
    returns the predicted author
    """
    count_vect = CountVectorizer(analyzer='word', lowercase=False, token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['text'])
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    author = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
    
    return author

def corpusMake(language):
    """
    This function creates a corpus using the files in a tirectory called text
    """
    # for each files in the directory, if they are from the texts folder, then dump it in the list
    # this function also auto sorts the list (hence why i had to make it text01, text02)
    if language == "english":
        files = [f for f in listdir('texts-10') if isfile(join('texts-10', f))]
    
        # then extract each files and we dump them in a list
        books = []
        for i in files:
            i = 'texts-10/' + i
            with open(i,'r', encoding='utf8', errors='ignore') as f:
                doc = f.readlines()
            books.append(doc)
            doc = ''
    elif language == "spanish":
        files = [f for f in listdir('tests-spanish') if isfile(join('tests-spanish', f))]
        
        books = []
        for i in files:
            i = 'tests-spanish/' + i
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

def dataset_prep(file, language):
    """
    this will prep the texts and author for later usage
    returns a corpus, the contents of user input in a form of corpus, and authors.
    """
    data = open('corpusdata.txt').read()
    labels, texts = [], []
    for i, line in enumerate(data.split("\n")):
        content = line.split()
        texts.append(content)
    if language == "english":
        with open('authors/authors.txt', 'r') as f:
            auth = f.readlines()
    elif language == "spanish":
        with open('authors-spanish/authors-spanish.txt', 'r') as f:
            auth = f.readlines()

    for i in range(len(auth)):
        labels.append(auth[i].split("\n"))    
        
    target_data = []
    for i in range(len(auth)):
        target_data.append(labels[i][0])
    
    temp = []
    with open("%s" % file, 'r', encoding='utf8', errors='ignore') as f:
        file_content = f.readlines()
        content = file_content[0].split()
        temp.append(content)
    
    # create a dataframe using texts and lables
    trainDF = pandas.DataFrame()
    trainDF['label'] = target_data
    texts.pop()
    trainDF['text'] = texts
    trainDF['text']=[" ".join(review) for review in trainDF['text'].values]
    
    UInput = pandas.DataFrame()
    UInput['text'] = temp
    UInput['text'] = [" ".join(review) for review in UInput['text'].values]
    return trainDF, UInput, target_data

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    """
    this function accepts any classifiers and predicts an author.
    returns the predicted author.
    """
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return predictions

if __name__ == '__main__':
    app.run(debug=True)
