import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pickle


df=pd.read_csv('Bank Customer Churn Prediction.csv')
df.head()

lb1 = LabelEncoder()
lb2 = LabelEncoder()


df['gender']= lb1.fit_transform(df[['gender']])
df['country'] = lb2.fit_transform(df[['country']])

df.head()
x=df.drop(['customer_id','churn'],axis=1)
y=df['churn']


from imblearn.under_sampling import NearMiss
nm=NearMiss(n_neighbors=5,sampling_strategy='majority',version=1)
x_resampled,y_resampled=nm.fit_resample(x,y)

from collections import Counter
# print('Original dataset shape %s' % Counter(y))
# print('Resampled dataset shape %s' % Counter(y_resampled))


xtrain,xtest,ytrain,ytest=train_test_split(x_resampled,y_resampled,test_size=.25,random_state=42)
# print(xtrain.shape)
# print(xtest.shape)
# print(ytrain.shape)
# print(ytest.shape)

ytest=ytest.values.reshape(-1,1)



scaler=StandardScaler()
xtrain=scaler.fit_transform(xtrain)
xtest=scaler.transform(xtest)

# print(xtest)


model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),n_estimators=50,learning_rate=0.01,random_state=42)
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
accuracy=model.score(xtest,ytest)
print(accuracy)



preprocessor = {
    'model':model,
    'label_encoder1': lb1,
    'label_encoder2': lb2,
    'scaler': scaler
}

file_path = 'D:/jupyter/practice/model.pkl'

try:
    with open('model.pkl', 'wb') as file:
        pickle.dump(preprocessor, file)
    print(f'Successfully saved model data to {file_path}')
except Exception as e:
    print(f"An error occurred: {str(e)}")

with open('model.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# xtest.to_csv('xtest.csv')
