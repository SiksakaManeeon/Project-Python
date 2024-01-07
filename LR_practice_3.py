import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

data = sklearn.datasets.load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(X.head())
print(X.tail())
print(X.info())
print(X.describe().round(1))
print(X.value_counts())
print(pd.DataFrame(y).describe())
X.hist(figsize=(30, 30), bins=30)
plt.show()
sns.histplot(data=y)
plt.show()
print(X.isna().sum())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=True,random_state=100)
print(len(X_train))
print(len(X_test))

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

train = X_train.copy()
train['target'] = y_train
print(train.head())

print(train.corr().round(2).mask(train.corr().abs()<0.5,''))
print(train.corr()['target'].abs().sort_values(ascending=False))
selectedfeatures = ['bmi', 's5']
print(X_train[selectedfeatures].corr(), '\n')

lr1 = SGDRegressor(learning_rate='constant',eta0=0.02,penalty=None,random_state=100,max_iter=10000)
lr1.fit(X_train[selectedfeatures], y_train)

y_train_lr1=pd.DataFrame(lr1.predict(X_train[selectedfeatures]))
y_test_lr1=pd.DataFrame(lr1.predict(X_test[selectedfeatures]))

fig,(ax1, ax2)=plt.subplots(2,1,figsize=(5,10))
ax1.scatter(y_train,y_train_lr1)
ax1.set_xlabel('Actual y_train')
ax1.set_ylabel('Predicted y_train')
ax1.set_title('scatter plot between actual y_train and predicted y_train')
ax2.scatter(y_test,y_test_lr1)
ax2.set_xlabel('Actual y_test')
ax2.set_ylabel('Predicted y_test')
ax2.set_title('scatter plot between actual y_test and predicted y_test')
plt.show()

print('mae :',metrics.mean_absolute_error(y_test, y_test_lr1))
print('mse :',metrics.mean_squared_error(y_test, y_test_lr1))
print('mape :',metrics.mean_absolute_percentage_error(y_test, y_test_lr1))
print('r2 :',metrics.r2_score(y_test, y_test_lr1))
print('Adjusted r2 :',1-(1-metrics.r2_score(y_test, y_test_lr1))*(len(X_test)-1)/(len(X_test)-13-1))

print(np.sqrt(metrics.mean_squared_error(y_test, y_test_lr1)))
print(np.mean(np.abs(y_test-np.array(y_test_lr1).flatten())/y_test))

coef_selected=pd.DataFrame(X_train[selectedfeatures].columns,columns=['features'])
coef_selected['coefficents']=lr1.coef_
coef_selected.loc[len(coef_selected)]=['INTERCEPT',lr1.intercept_[0]]
print(coef_selected)