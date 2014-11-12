from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib as mpl
train=pd.read_csv("train.csv")

indext=pd.DatetimeIndex(train['datetime'].values)
train=train.reset_index(indext)
train['hour']=indext.hour
train['date']=indext.date
train['month']=indext.month
train['year']=indext.year
train['dayofweek']=indext.dayofweek
train=train.dropna()
small=train[['registered','hour','dayofweek','season','atemp','humidity','holiday','windspeed','temp']]
for i in range(4):
    small['season'+str(i+1)]=train['season']==i+1
small['year11']=train['year']=2011
for i in range(7):
    small['day'+str(i+1)]=train['dayofweek']==i+1
small['hour5-10']=[1 if i<10 & i>=5 else 0 for i in train['hour'] ]
small['hour10-15']=[1 if i<15 & i>=10 else 0 for i in train['hour'] ]
small['hour15-20']=[1 if i<20 & i>=15 else 0 for i in train['hour'] ]
small['hour>20']=[1 if i>=20 else 0 for i in train['hour'] ]
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
Y=train[['registered']].values
X=train[['hour','month','year']].values
x_train, x_test,y_train,y_test =train_test_split(X,Y)
y_train=y_train.ravel()
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
print rf.score(x_train,y_train),rf.score(x_test,y_test)