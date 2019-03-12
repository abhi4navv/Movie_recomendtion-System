from flask import Flask,render_template
from flask import render_template
from flask import request
from bs4 import BeautifulSoup
from urllib.request import Request,urlopen
import ssl
import re
import pandas as pd
import sqlite3
import pyodbc
import json
import requests

from nltk.corpus import stopwords
from nltk.stem.porter import*
import string
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
import pickle
from sklearn import tree, metrics, model_selection , preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,BaggingRegressor,RandomForestRegressor
from xgboost.sklearn import XGBClassifier
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models.doc2vec import LabeledSentence
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def check_Movie_name(Movie_Name):
    con=sqlite3.connect('C:\\Users\\abhinav\\Desktop\\Data sacience class data\\Movie_Review.db')
    cur=con.cursor()
    con.execute("PRAGMA busy_timeout=30000")
    sqlstr="SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    cur.execute(sqlstr,('Movie_Reviews',))
    cur.close()
    if cur.rowcount==-1:
        json_data=getdata(Movie_Name)

    else:
        con.execute("PRAGMA busy_timeout=30000")
        sqlstr2='SELECT Movie_Name,Movie_Reviews FROM Movie_Reviews WHERE Movie_Name LIKE ? LIMIT(5)'
        cur.execute(sqlstr2,(Movie_Name+'%',))

        if cur.rowcount==-1:
             json_data=getdata(Movie_Name)
        else:
            json_data=get_revs(Movie_Name,'show')
        cur.close()
    return json_data

def getdata(Movie_Name):
    req=Request('https://www.rottentomatoes.com/m/'+Movie_Name,headers={'User-Agent':'Mozilla/5.0'})
    try:
        htmldt=urlopen(req).read()
    except:
        htmldt='0'
    if htmldt !='0':
        soup=BeautifulSoup(htmldt,'html.parser')
        list1={}
        list2=list()
        for div in soup.findAll('div',{'class':'review_quote'}):
            list1[div.find('p')] = div.text.strip().replace('','')
        for dt in list1:
            rev=re.findall('\s([a-zA-Z].*)',str(dt))
            list2.append(rev)
        save_rev_to_db(list2,Movie_Name)
        json_data=get_revs(Movie_Name)
        dfrev=get_movierev_data(json_data)
        dfcl=remove_stopword(dfrev['Movie_Review'])
        dfflt= rem_shortword(dfcl)
        dfflt=replace_empty_rows(dfflt)
        dfflt=drop_na(dfflt)
        rev_tokens=create_wordtokens(dfflt)
        dfrev['Filt_revstm']=dfflt.apply(stem_words)
        dfrev['Filt_revlem']=dfflt.apply(lem_words)
        bow_stem=get_bag_of_words(dfrev['Filt_revstm'],'bag_of_words_stem.pkl')
        bow_lem=get_bag_of_words(dfrev['Filt_revlem'],'bag_of_words_lem.pkl')
        df_Results=pd.read_csv('df_result.csv')
        df_Results.columns=['Model_Type','Accuracy','Recall','Precision','F1_Score','Model_FileName']

        fname_naive_BagWord_stem=get_file_names('Naive_bayes_With_stem_Bag_words',df_Results)
        pred_test_Naive_BagWord_stem=get_pred_test_data(fname_naive_BagWord_stem,bow_stem)
        fname_naive_BagWord_lem=get_file_names('Naive_bayes_With_lem_Bag_words',df_Results)
        pred_test_Naive_BagWord_lem=get_pred_test_data(fname_naive_BagWord_lem,bow_stem)
        tfwords_test_stem=get_tfidf_data(dfrev['Filt_revstm'],'TFIDF_stem.pkl')
        tfwords_test_lem=get_tfidf_data(dfrev['Filt_revlem'],'TFIDF_lem.pkl')
        fname_naive_stem_TFIDF=get_file_names('Naive_bayes_With_stem_TFIDF',df_Results)
        pred_test_Naive_stem_TFIDF=get_pred_test_data(fname_naive_stem_TFIDF,tfwords_test_stem)
        fname_naive_lem_TFIDF=get_file_names('Naive_bayes_With_lem_TFIDF',df_Results)
        pred_test_Naive_lem_TFIDF=get_pred_test_data(fname_naive_lem_TFIDF,tfwords_test_lem)
        dfrev['pred_test_Naive_BagWord_stem']=pred_test_Naive_BagWord_stem
        dfrev['pred_test_Naive_BagWord_lem']=pred_test_Naive_BagWord_lem
        dfrev['pred_test_Naive_stem_TFIDF']=pred_test_Naive_stem_TFIDF
        dfrev['pred_test_Naive_lem_TFIDF']=pred_test_Naive_lem_TFIDF
        save_Predicted_csv(dfrev,'PredictedData.csv')
        save_predicted_values('PredictedData.csv')
        json_data=get_revs(Movie_Name,'show')
    else:
        json_data='0'
    return json_data


def save_rev_to_db(rev_list,Movie_Name):
    con=sqlite3.connect('C:\\Users\\abhinav\\Desktop\\Data sacience class data\\Movie_Review.db')
    cur=con.cursor()
    lst_rev=list()
    for dt in rev_list:
        for revs in dt:
            lst_rev.append(str(revs))
    df=pd.DataFrame()
    df['Reviews']=lst_rev
    df.drop_duplicates(subset='Reviews',keep='first',inplace=True)
    cur= con.cursor()
    con.execute("PRAGMA busy_timeout=30000")
    cur.execute('CREATE TABLE IF NOT EXISTS Movie_Reviews (Id INTEGER PRIMARY KEY,Movie_Name TEXT,Movie_Reviews TEXT,Movie_Ratings TEXT,pred_test_Naive_BagWord_stem TEXT,pred_test_Naive_BagWord_lem TEXT,pred_test_Naive_stem_TFIDF TEXT,pred_test_Naive_lem_TFIDF TEXT)')
    for rev in df['Reviews']:
        cur.execute('INSERT INTO Movie_Reviews(Movie_Name,Movie_Reviews,Movie_Ratings,pred_test_Naive_BagWord_stem,pred_test_Naive_BagWord_lem,pred_test_Naive_stem_TFIDF,pred_test_Naive_lem_TFIDF) VALUES(?,?,?,?,?,?,?)',(Movie_Name,rev,'NULL','NULL','NULL','NULL','NULL',))
    con.commit()
    cur.close()

def get_revs(Movie_Name,reqtype='getdata'):
    con=sqlite3.connect('C:\\Users\\abhinav\\Desktop\\Data sacience class data\\Movie_Review.db')
    cur=con.cursor()
    cur = con.cursor()
    con.execute("PRAGMA busy_timeout=30000")
    if reqtype=='show':
        sqlstr='SELECT Id,Movie_Name,Movie_Reviews,pred_test_Naive_BagWord_stem,pred_test_Naive_BagWord_lem,pred_test_Naive_stem_TFIDF,pred_test_Naive_lem_TFIDF FROM Movie_Reviews WHERE Movie_Name=? LIMIT(10)'
    else:
        sqlstr='SELECT Id,Movie_Name,Movie_Reviews,pred_test_Naive_BagWord_stem,pred_test_Naive_BagWord_lem,pred_test_Naive_stem_TFIDF,pred_test_Naive_lem_TFIDF FROM Movie_Reviews WHERE Movie_Name=? '
    cur.execute(sqlstr,(Movie_Name,))
    if cur.rowcount==0:
        json_data='0'
    else:
        json_data=cur.fetchall()
    cur.close()
    return json_data

#js_data=check_Movie_name('how_to_train_your_dragon_the_hidden_world')

def get_movierev_data(js_data):
    lstname=list()
    lstrev=list()
    lstid=list()

    clmns=['Id','Movie_Name','Movie_Review']
    dfrev=pd.DataFrame(columns=clmns)
    dfrev.columns=clmns
    for dt2 in js_data:
            lstid.append(dt2[0])
            lstname.append(dt2[1])
            lstrev.append(dt2[2])
    dfrev['Id']=lstid
    dfrev['Movie_Name']=lstname
    dfrev['Movie_Review']=lstrev
    return dfrev

def remove_stopword(rawdata):
    stop=stopwords.words('english')
    punc=string.punctuation
    dfcnl=rawdata.apply(lambda x:' '.join([item for item in x.split() if item  not in stop]))
    dfcnl=dfcnl.apply(lambda x: ' '.join([wrd for wrd in x.split() if wrd not in punc]))
    return dfcnl

def rem_shortword(textdata):
    filtdata=textdata.apply(lambda x: ' '.join([wd for wd in x.split() if len(wd)>3]))
    return filtdata

def replace_empty_rows(dfr):
    dfr.replace('',np.nan,inplace=True)
    return dfr


def drop_na(dfrs):
    dfrs.dropna(inplace=True)
    return dfrs

def create_wordtokens(sent):
    word_tokens=sent.apply(lambda x: x.split())
    return word_tokens

def stem_words(sent):
    tokens= str(sent).split()
    stemmer=PorterStemmer()
    stemed_dt= [stemmer.stem(token) for token in tokens]
    stemed_dt=' '.join(stemed_dt)
    return stemed_dt
def lem_words(sents):
    tokens=str(sents).split()
    lemmatizer=WordNetLemmatizer()
    lem_word=[lemmatizer.lemmatize(token) for token in  tokens]
    lem_word=' '.join(lem_word)
    return lem_word

def get_bag_of_words(filtered_data,fname):
    vectorize=CountVectorizer()
    loaded_vec = CountVectorizer(vocabulary=pickle.load(open(fname, "rb")))
    bow=loaded_vec.fit_transform(filtered_data)
    return bow


def get_tfidf_data(input_data,fname):
    tfvec=TfidfVectorizer(max_df=0.9,min_df=0.0,max_features=1000,stop_words='english')
    loaded_vec = CountVectorizer(vocabulary=pickle.load(open(fname, "rb")))
    tfidfdt=loaded_vec.fit_transform(input_data)
    return tfidfdt
def save_pred_data(files,ids,fnames):
    df_filedata=pd.DataFrame()
    df_filedata['Id']=ids
    df_filedata['Pred_Data']=files
    df_filedata.to_csv(fnames,index=False)
def get_pred_test_data(model_file,feat_words):
    model_name=get_model(model_file)
    pred_test=model_name.predict(feat_words)
    return pred_test

def get_file_names(model,dfs):
    fnames=dfs[dfs['Model_Type']==model]['Model_FileName']
    for dt1 in fnames:
        fname=dt1
    return fname



def get_model(filename):
    model=pickle.load(open(filename,'rb'))
    return model

def save_Predicted_csv(df_Preds,fnames):
    df_Preds.to_csv(fnames,index=False)

def save_predicted_values(df_predictedfname):
    df_predicted=pd.read_csv(df_predictedfname)
    con=sqlite3.connect('C:\\Users\\abhinav\\Desktop\\Data sacience class data\\Movie_Review.db')
    cur=con.cursor()
    cur = con.cursor()
    for i,row in df_predicted.iterrows():
        id=row['Id']
        pred_Naive_BagWord_stem=row['pred_test_Naive_BagWord_stem']
        pred_Naive_BagWord_lem=row['pred_test_Naive_BagWord_lem']
        pred_Naive_stem_TFIDF=row['pred_test_Naive_stem_TFIDF']
        pred_Naive_lem_TFIDF=row['pred_test_Naive_lem_TFIDF']
        sqlstr2='''UPDATE Movie_Reviews SET pred_test_Naive_BagWord_stem=?,pred_test_Naive_BagWord_lem=?,
        pred_test_Naive_stem_TFIDF=?,pred_test_Naive_lem_TFIDF=? WHERE Id=?'''
        cur.execute(sqlstr2,(pred_Naive_BagWord_stem,pred_Naive_BagWord_lem,pred_Naive_stem_TFIDF,pred_Naive_lem_TFIDF,id))
    con.commit()
    cur.close()

def get_suggestion(Movie_Name):
    con=sqlite3.connect('C:\\Users\\abhinav\\Desktop\\Data sacience class data\\Movie_Review.db')
    movie=Movie_Name.replace('_',' ')
    movie=movie.upper()
    sqlstr=''
    cntgood=0
    cntbad=0
    percentgood=0.0
    percentbad=0.0
    difgood=0.0
    difbad=0.0
    messages=''
    cur=con.cursor()
    cur = con.cursor()
    con.execute("PRAGMA busy_timeout=30000")
    sqlstr='''SELECT
    ( SELECT COUNT(Id) FROM Movie_Reviews where pred_test_Naive_BagWord_stem ="Good" AND Movie_Name=?) AS "Good Reviews",
    (SELECT COUNT(ID) FROM Movie_Reviews WHERE pred_test_Naive_BagWord_stem="Bad" AND Movie_Name=?) As "Bad Reviews" '''
    cur.execute(sqlstr,(Movie_Name,Movie_Name,))
    if cur.rowcount==0:
        messages=''
    else:
        json_data=cur.fetchall()
        cntgood=int(json_data[0][0])
        cntbad=int(json_data[0][1])

        if cntgood==cntbad:
                messages= movie +' seems to be an average movie as number og good reviews are same as number of bad reviews  i.e. ' + str(cntgood)
        else:
            percentgood=((cntgood/(cntgood+cntbad))*100)
            percentbad=((cntbad/(cntgood+cntbad))*100)
            if percentgood>percentbad:
                difgood=percentgood-percentbad
                if difgood>5.:
                    messages= +movie + 'is good movie because we have '+ str(difgood)+'% more good reviews than bad reviews.'
                else:
                    messages=movie + ' seems to be an average movie because we have only '+ str(difgood)+'% difference between good reviews and bad reviews.'

            elif percentbad>percentgood:
                difbad=percentbad-percentgood
                if difbad>5.:
                    messages='It seems that  '+movie + ' has not impressed many audience because we have '+ str(difbad)+'% more bad reviews than good reviews.'
                else:
                    messages=movie + ' seems to be an average movie because we have only '+ str(difbad) +'% difference between good reviews and bad reviews.'
            else:
                messages='Oops the number of good and bad reviews have confused us so please watch the movie and provide other viewers a better review.'

    cur.close()
    return messages

# Flask for making a UI for the application.
app=Flask(__name__)
@app.route("/home",methods=["GET","POST"])
def home():
    json_datafin=''
    json_headers=[]
    message=''
    messages=''
    if request.method=='POST':
        text=request.form['text']
        text=text.lower()
        text=str(text).replace(' ','_')
        json_datafin=check_Movie_name(text)
        if json_datafin=='0':
            message='oops either the movie name is incorrect or there is no reviews for movie yet please watch it own your own risk...'
            json_headers=[]
            messages=''
        else:
            messages=get_suggestion(text)
            json_headers=['Movie Name','Movie Reviews','Prediction using Naive Bag of Word and stemming','Prediction using Naive Bag of Word and lemmatization','Prediction using Naive TFIDF and lemmatization','Prediction using Naive TFIDF and lemmatization']
    return render_template('home.html',posts=json_datafin,message=message,json_headers=json_headers,messages=messages)
        #else:
            #return render_template('home.html',revs=json_datafin)

if __name__=='__main__':
    app.run(debug=True)
