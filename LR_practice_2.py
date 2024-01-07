import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier,export_text,plot_tree
from sklearn import metrics
from yellowbrick.classifier import ROCAUC
from yellowbrick.model_selection import ValidationCurve
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()
print(iris.target_names)
print(iris.feature_names)

data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = iris.target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3,shuffle=True,stratify=target,random_state=100)

print(X_train.corr().round(2).mask((X_train.corr()<=0.9)&(X_train.corr()>=-0.9),''))

NB = GaussianNB()
NB.fit(X_train,y_train)
y_pred_1 = NB.predict(X_test)

print("Classification Report: \n", metrics.classification_report(y_test,y_pred_1,target_names=iris.target_names))
print("Accuracy on train:  ",NB.score(X_train,y_train).round(3))
print("Accuracy on test: ",NB.score(X_test,y_test).round(3))
cf = metrics.confusion_matrix(y_test,y_pred_1).round(3)
sns.heatmap(cf, annot=True,cmap='Blues',yticklabels=iris.target_names,xticklabels=iris.target_names)
plt.ylabel("True Label")
plt.xlabel("Predict Label")
plt.show()

y_pred_1_prob=NB.predict_proba(X_test)
metrics.roc_auc_score(y_test,y_pred_1_prob,multi_class='ovr')

visualizer = ROCAUC(NB, classes=iris.target_names)
visualizer.fit(X_train, y_train)      
visualizer.score(X_test, y_test) 
visualizer.show()

################################### NB #############################

DT = DecisionTreeClassifier(random_state=100)
cv = StratifiedKFold(5)
param_val = [{'criterion':['entropy','gini'],'max_depth':[1,5],'min_samples_split':np.arange(2,10,2),'ccp_alpha':[0.01,0.05,0.1]}]
grid = GridSearchCV(DT, param_val, cv = cv,scoring='accuracy')
grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_params_)

y_pred_2 = grid.predict(X_test)

grid.predict_proba(X_test)

print("Classification Report: \n", metrics.classification_report(y_test,y_pred_2,target_names=iris.target_names))
print("Accuracy on train:  ",grid.score(X_train,y_train).round(3))
print("Accuracy on test: ",grid.score(X_test,y_test).round(3))
cf=metrics.confusion_matrix(y_test,y_pred_2).round(3)
sns.heatmap(cf, annot=True,cmap='Blues',yticklabels=iris.target_names,xticklabels=iris.target_names)
plt.ylabel("True Label")
plt.xlabel("Predict Label")
plt.show()

y_pred_2_prob = grid.predict_proba(X_test)
metrics.roc_auc_score(y_test,y_pred_2_prob,multi_class='ovr')

visualizer = ROCAUC(grid, classes=iris.target_names)
visualizer.fit(X_train, y_train)       
visualizer.score(X_test,y_test)
visualizer.show()

print(export_text(grid.best_estimator_,feature_names=X_test.columns.tolist()))

plt.figure(figsize=(12,12),dpi=100)
plot_tree(grid.best_estimator_,feature_names=X_test.columns,class_names=iris.target_names)
plt.show()

################################### DT #############################