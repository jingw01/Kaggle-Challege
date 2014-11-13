from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib as mpl
from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split

def normalize(train,test):
    norm=preprocessing.Normalizer()
    train=norm.fit_transform(train)
    test=norm.transform(test)
    return train,test

train=pd.read_csv("train.csv")

indext=pd.DatetimeIndex(train['datetime'].values)
train=train.reset_index(indext)
train['hour']=indext.hour
train['date']=indext.date
train['month']=indext.month
train['year']=indext.year
train['dayofweek']=indext.dayofweek


def createDecisionTree():
    est = DecisionTreeRegressor()
    return est

def createRandomForest():
    est = RandomForestRegressor(n_estimators=500)
    return est

def createExtraTree():
    est = ExtraTreesRegressor(n_estimators=700)
    return est

def createGradientBoostingRegressor():
    est = GradientBoostingRegressor()
    return est

def createKNN():
    est = KNeighborsRegressor(n_neighbors=2)
    return est

Y=train[['count']].values
X=train[['hour','season','year','temp','holiday','windspeed','humidity','weather']].values
x_train, x_test,y_train,y_test =train_test_split(X,Y)
y_train=y_train.ravel()
rf=createDecisionTree()
rf.fit(x_train,y_train)
print 'DecisionTres', rf.score(x_train,y_train),rf.score(x_test,y_test)
rf=createExtraTree()
rf.fit(x_train,y_train)
print 'ExtraTree', rf.score(x_train,y_train),rf.score(x_test,y_test)
rf=createGradientBoostingRegressor()
rf.fit(x_train,y_train)
print 'GadientBoosting', rf.score(x_train,y_train),rf.score(x_test,y_test)
rf=createKNN()
rf.fit(x_train,y_train)
print 'KNN', rf.score(x_train,y_train),rf.score(x_test,y_test)

