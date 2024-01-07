import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.classifier import ROCAUC

IRIS = datasets.load_iris()

data_Iris = pd.DataFrame(IRIS.data, columns=IRIS.feature_names)
target = pd.DataFrame(IRIS.target, columns=['class'])
target_1 = IRIS.target

data_Iris_1 = data_Iris.copy()

normalizer=MinMaxScaler().fit(data_Iris_1)
data_Iris_1=pd.DataFrame(normalizer.transform(data_Iris_1), columns=IRIS.feature_names)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data_Iris_1,target)
target_pred = knn.predict(data_Iris_1)

print("Classification Report: \n", metrics.classification_report(target,target_pred,target_names=IRIS.target_names),'\n')
print("Accuracy: ",metrics.accuracy_score(target,target_pred),'\n')
cf=metrics.confusion_matrix(y_true=target,y_pred=target_pred).round(3)
sns.heatmap(cf,annot=True,cmap='Blues')
plt.ylabel("True Label")
plt.xlabel("Predict Label")
plt.show()

visualizer = ROCAUC(knn, classes=IRIS.target_names)
visualizer.fit(data_Iris_1, target)   
visualizer.score(data_Iris_1,target)      
visualizer.show()