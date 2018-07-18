# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 10:40:07 2018


@author: Dell
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import easygui
import os
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.metrics import accuracy_score,classification_report,mean_absolute_error,mean_squared_error

cdir=os.getcwd()
def get_data():
    filepath=easygui.fileopenbox(msg='Provide a Training data',title='Getting data',filetypes='csv')
    print(filepath)
    df=pd.read_csv(filepath,encoding="ISO-8859-1")
    return df

global data_df
data_df=get_data()
global columns
columns=data_df.columns
def choose_features():
    columns=data_df.columns
    choosen_choice=easygui.multchoicebox(msg='Choose features for training',title='Feature Engineering',choices=columns)
    print(choosen_choice)
    return choosen_choice

#choose_features()

def choose_target():
    target=easygui.choicebox(msg='choose target to predict',title='Ground truth selection',choices=columns)
    print(target)
    return target
#pd.read_csv(r'E:\python files\wrist watch_amazon.csv',encoding="ISO-8859-1")

#choose_target()
def analyse_data():
    try:
        import dfgui
        dfgui.show(data_df)
    except:
        print('Error in analysing library')
    


def choose_algorithm():
    prb_type=['Classification','Regression']
    alg_type=easygui.choicebox(msg='Choose problem type',title='Classification or Regression',choices=prb_type)
    if alg_type=='Classification':
        algorithms=['Logistic Regression','Gradient Boosting Classifier','Random forest classifier']
        choosen_alg=easygui.choicebox(msg='Choose Algorithm you want to apply',title='Choosing '+str(alg_type)+' Algorithms',choices=algorithms)
    if alg_type=='Regression':
        algorithms=['Linear Regression','Gradient Boosting Regressor','Random forest Regressor']
        choosen_alg=easygui.choicebox(msg='Choose Algorithm you want to apply',title='Choosing '+str(alg_type)+' Algorithms',choices=algorithms)
    print(choosen_alg)
    return choosen_alg,alg_type
        
#choose_algorithm()    

def Train_test_split():
    yn=easygui.ynbox(msg='Do you want to split your data into train and split',title='Train Test Split')
    if yn:
        splt=str(easygui.choicebox(msg='Pick the percentage split for train and test split',title='Train test Percentage',choices=['66-34','75-25','80-20']))
        train_per=int(splt.split('-')[0])
        return train_per
    else :
        pass
#        easygui
        
    
def apply_algorithm():
    analyse_data()
    features=choose_features()
    target=choose_target()
    alg,alg_type=choose_algorithm()
    filtered_df=data_df[features]
    target=data_df[target]
    train_per=Train_test_split()
    X_train,X_test,y_train,y_test=train_test_split(filtered_df,target,train_size=train_per)
    if alg=='Linear Regression':
        clf=LinearRegression()
    if alg=='Gradient Boosting Regressor':
        clf=GradientBoostingRegressor()
    if alg=='Random forest Regressor':
        clf=RandomForestRegressor()
    if alg=='Logistic Regression':
        clf=LogisticRegression()
    if alg=='Gradient Boosting Classifier':
        clf=GradientBoostingClassifier()
    if alg=='Random forest classifier':
        clf=RandomForestClassifier()
#    if alg=='Linear Regression':
#        pass
#    if alg=='Linear Regression':
#        pass
    clf.fit(X_train,y_train)
    predicted=clf.predict(X_test)
    if alg_type=='Classification':
        auc=accuracy_score(y_test,predicted)
        cr=classification_report(y_test,predicted)
        easygui.msgbox(msg='Your accuracy {}'.format(auc))
        easygui.msgbox(msg='Classification report {}'.format(cr))
        #easygui.msgbox(msg='Classi'
        print(auc,cr)
    if alg_type=='Regression':
        mae=mean_absolute_error(y_test,predicted)
        mse=mean_squared_error(y_test,predicted)
        rmse=mse**0.5
        easygui.msgbox(msg='Your Mean Absolute error {}'.format(mae))
        easygui.msgbox(msg='Your Mean Square error  {}'.format(mse))
        easygui.msgbox(msg='Your Root Mean square error {}'.format(rmse))
        print(mae,mse,rmse)
    out_df=pd.DataFrame()
    out_df['Actual']=y_test
    out_df['Predicted']=predicted
    cdir=os.getcwd()+'\TEST'
    if not os.path.exists(cdir):
        os.mkdir(cdir)
    
    cdir=cdir+'\Test.csv'    
    out_df.to_csv(cdir)
    
    
    

#if __name__=='__main__':
apply_algorithm()




#from sklearn.datasets import make_regression,make_classification
#r=make_regression(n_samples=100,n_features=8)
#s=make_classification(n_classes=2,n_samples=100,n_features=8)
#s[0]
#s[1]
#
#dummy_df=pd.DataFrame(s[0],columns=[i for i in range(0,8)])
#dummy_df['target']=s[1]
#dummy_df.to_csv(r'E:\python files\PROML\class_ex.csv')
#
#
#dummy_df=pd.DataFrame(r[0],columns=[i for i in range(0,8)])
#dummy_df['target']=r[1]
#dummy_df.to_csv(r'E:\python files\PROML\reg_ex.csv')
#






