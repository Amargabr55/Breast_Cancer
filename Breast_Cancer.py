import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import pickle
import json
#recall data by using pandas
datasets = pd.read_csv("Breast_Cancer.csv")
#split object data and num data t convert object data to numbers to using in algorithm
object_data = datasets.select_dtypes(include=["object"])

le = LabelEncoder()
for i in range(object_data.shape[1]):
    object_data.iloc[:, i] = le.fit_transform(object_data.iloc[:, i])
    num_data = datasets.select_dtypes(exclude=["object"])
new_datasets = pd.concat([object_data, num_data], axis=1)
#split data
x1 = new_datasets.iloc[:, :9]
x2 = new_datasets.iloc[:, 11:]

#concationtion data 
x = pd.concat([x1,x2], axis=1)
# split y_actual or actual output
y = new_datasets.iloc[:, 10]
#create list ti use in pairplot
m_col = ['Tumor_Size','N Stage','Age','Grade']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
#convert data from classfication to regression 
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

#Created pair plot in order to clarify data details
pairplot_data = pd.concat([new_datasets[m_col], pd.Series(y_train_encoded, name='Predicted T Stage')], axis=1)
sns.pairplot(pairplot_data, hue='Predicted T Stage', palette='husl')
plt.show()
#implement algorithm 
svc = SVC(kernel='linear', C = 0.1 )   
svc.fit(x_train, y_train_encoded) 
y_pred_encoded = svc.predict(x_test)
#the show result algorithm 
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)

print('Accuracy:', accuracy)
#create confusion matrix to compare between actual data and predicted data
cm=metrics.confusion_matrix(y_test_encoded, y_pred_encoded)
print(cm)
pickle.dump(svc,open('breast_cancer.pkl', 'wb'))
model=pickle.load(open('breast_cancer.pkl','rb'))

print(model.predict([[2,1,1,2,0,2,1,1,1,50,35,14,5,62,1]]))
print(model.predict([[2,0,2,4,0,2,1,1,1,58,63,14,5,75,0]]))

with open('breast_cancer.pkl', 'wb') as f:
    pickle.dump((svc, label_encoder), f)