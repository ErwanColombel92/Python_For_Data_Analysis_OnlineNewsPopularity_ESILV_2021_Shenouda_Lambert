

from flask import Flask 
from flask import render_template, redirect, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, InputRequired, Length, NumberRange
import joblib
import sklearn
import config
import xgboost

############ RAJOUT ###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from newspaper import Article
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score

from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stopwords=set(stopwords.words('english'))
from textblob import TextBlob #for subjectivity and polarity purpose

def tokenizetext(text):
    return word_tokenize(text)
def words(text):
    l = [word for word in word_tokenize(text) if word.isalpha()]
    return l
def unique_words(text):
    return list(set(words(text)))
def rate_uni_words(text):
    uni_words = len(unique_words(text))/len(words(text))
    return uni_words
def avglengthtoken(text):
    w = words(text)
    sum = 0
    for item in w:
        sum+=len(item)
    avglen = sum/len(w)
    return avglen
def n_non_stop_unique_tokens(text):
    uw = unique_words(text)
    n_uw = [item for item in uw if item not in stopwords]
    w = words(text)
    n_w = [item for item in w if item not in stopwords]
    rate_nsut = len(n_uw)/len(n_w)
    return rate_nsut
def numlinks(article):
    return len(BeautifulSoup(article).findAll('link'))
def get_subjectivity(a_text):
    return a_text.sentiment.subjectivity
def get_polarity(a_text):
    return a_text.sentiment.polarity
def word_polarity(words):
    pos_words = []
    ppos_words = [] # polarity of pos words
    neg_words = []
    pneg_words = [] # polarity of negative words
    neu_words = []
    pneu_words = [] # polarity of neutral words
    for w in words:
        an_word = TextBlob(w)
        val = an_word.sentiment.polarity
        if val > 0:
            pos_words.append(w)
            ppos_words.append(val)
        if val < 0:
            neg_words.append(w)
            pneg_words.append(val)
        if val == 0 :
            neu_words.append(w)
            pneu_words.append(val)
    return pos_words,ppos_words,neg_words,pneg_words,neu_words,pneu_words
def avg_pol_pw(text):    
    totalwords = words(text)
    res = word_polarity(totalwords)
    return np.sum(res[1])/len(res[0])
def avg_pol_nw(text):    
    totalwords = words(text)
    res = word_polarity(totalwords)
    return np.sum(res[3])/len(res[2])


import nltk
nltk.download('punkt')
def masterDF(titre, texte, categ):
    finrows = []
    row = {}
    row['n_tokens_title'] = len(words(titre))
    row['n_tokens_content'] = len(words(texte))
    row['n_unique_tokens'] = len(unique_words(texte))
    row['average_token_length'] = avglengthtoken(texte)
    row['n_non_stop_unique_tokens'] = n_non_stop_unique_tokens(texte)
    row['num_hrefs'] = numlinks(texte)
    
    analysed_text = TextBlob(texte)
    row['global_subjectivity'] = get_subjectivity(analysed_text)
    row['avg_positive_polarity'] = avg_pol_pw(texte)
    row['global_sentiment_polarity'] = get_polarity(analysed_text)
    finrows.append(row)
    
    liste = ["data_channel_is_world", "data_channel_is_tech", "data_channel_is_socmed", "data_channel_is_bus", "data_channel_is_entertainment", "data_channel_is_lifestyle"]
    row["data_channel_is_world"] =0
    row["data_channel_is_tech"]=0
    row["data_channel_is_socmed"]=0
    row["data_channel_is_bus"]=0
    row["data_channel_is_entertainment"]=0
    row["data_channel_is_lifestyle"]=0
    row[liste[categ]] = 1
    
    
    return finrows

import pickle 
xgb2 = pickle.load(open("xgbmodel2", 'rb'))

def Prediction(titre, texte, categ):
    finrows = masterDF(titre, texte, categ)
    masterdf = pd.DataFrame(finrows, columns = ['n_tokens_title','n_tokens_content','n_unique_tokens','average_token_length','n_non_stop_unique_tokens','num_hrefs','global_subjectivity',
                                   'avg_positive_polarity','global_sentiment_polarity',
    'data_channel_is_world',"data_channel_is_tech", "data_channel_is_socmed","data_channel_is_bus","data_channel_is_entertainment","data_channel_is_lifestyle"])
    #0 world / 1 tech / 2 social / 3 business / 4 entertainement / 5 lifestyle
    pred = xgb2.predict(masterdf)
    pop = 'Popular' if pred == 1 else 'Unpopular'
    #print("Predicted popularity :",pop)
    return pop
    
#Exemple 1
print("Exemple Gab : ",Prediction("Emmanuel Macron: French president tests positive for Covid",
           "France's President Emmanuel Macron has tested positive for Covid-19 after developing symptoms.\
The 42 year old will now self-isolate for seven days, the Elysée Palace said in a statement.\
Mr Macron is still in charge of running the country and will work remotely, said an official.\
EU chief Charles Michel and Spanish Prime Minister Pedro Sánchez are both self-isolating after coming into contact with Mr Macron on Monday.\
France this week imposed an overnight curfew to help deal with soaring cases there.\
There have been two million confirmed cases in the country since the epidemic began, with more than 59,400 deaths, according to data from Johns Hopkins University.\
    How are France and other European countries tackling the pandemic?\
    French culture takes centre stage in Covid protest\
    The meteoric rise of France's youngest president\
Who has Macron had contact with?\
The President of the Republic has been diagnosed positive for Covid-19 today, the Elysée said on Thursday morning. The diagnosis was made following a test performed at the onset of the first symptoms, the statement added.\
It is not yet known how Mr Macron caught the virus but his office said it was identifying any close contacts he has made in recent days.\
Prime Minister Jean Castex, 55, and Parliament Speaker Richard Ferrand, 58, are both self-isolating, their offices confirmed.\
Mr Castex, who is not showing any symptoms, was due to introduce the government's Covid vaccination policy in the Senate on Thursday - now Health Minister Olivier Véran is doing it instead. ",
          0))
#0 world / 1 tech / 2 social / 3 business / 4 entertainement / 5 lifestyle

###################################################







#from flask_sqlalchemy import SQLAlchemy
#import os

#model_logistic = joblib.load("./xgbmodel2")
#test = model_logistic(['hello', 'test', 2])
#print (test)

app = Flask(__name__)
Bootstrap(app)
app.config['SECRET_KEY'] = 'hard to guess string'
app.config.from_object('config')
#db = SQLAlchemy(app)
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir,"data.sqlite")
#basedir = os.path.abspath(os.path.dirname(__file__))
#os.path.join(basedir,"data.sqlite")

#model_logistic = joblib.load("./xgbmodel2")
prediction = Prediction("hello", "test", 2)
print("Exemple Benj : Nous prédisons que l'article sera : " + prediction)
        
class EnterYourInfos(FlaskForm):
    title = StringField("Enter the article's title:", validators=[DataRequired()])
    text = StringField("Enter the article's text:", validators=[DataRequired()])
    aType = SelectField('Article Type', choices=['World', 'Tech', 'Social', 'Buisness', 'Entertainment', 'Lifestyle'])
    #aType = IntegerField("Article Type")
    submit = SubmitField("Submit")
   

@app.route('/', methods=["GET","POST"])
def show_user_the_form():
    myform= EnterYourInfos() 
    if request.method == "POST" and myform.validate_on_submit():
        model_logistic = joblib.load("./xgbmodel2")
        bType = 0;
        if myform.aType.data == "World":
            bType = 0; 
        if myform.aType.data == "Tech":
            bType = 1; 
        if myform.aType.data == "Social":
            bType = 2;
        if myform.aType.data == "Buisness":
            bType = 3; 
        if myform.aType.data == "Entertainment":
            bType = 4; 
        if myform.aType.data == "Lifestyle":
            bType =5; 
            
        prediction2 = str(Prediction(myform.title.data, myform.text.data, bType))
        if prediction2=='Popular':
            return render_template('result.html',**locals())
        if prediction2=='Unpopular':
            return render_template('unpop.html',**locals())
        
        
    
    return render_template('form.html',form=myform)


#@app.route('/result')
#def modelResponse():
#    return render_template('result.html')
    

#@app.route('/')
#def function():
#    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)


