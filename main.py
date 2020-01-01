import numpy as np # linea
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv('.data/Data-titanic/train.csv')
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
train['Age'] = train['Age'].fillna(np.mean(train['Age']))
train['Fare'] = train['Fare'].fillna(np.mean(train['Fare']))
train = train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
X = train.drop('Survived', axis = 1)
y = train['Survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
classifier=LogisticRegression()
classifier.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
print('Training accuracy...', accuracy_score(y_train, classifier.predict(X_train)))
print('Validation accuracy', accuracy_score(y_test, classifier.predict(X_test)))

import pickle
pickle.dump(classifier, open('.artifacts/model.pickle', 'wb'))

results=[accuracy_score(y_train, classifier.predict(X_train)),accuracy_score(y_test, classifier.predict(X_test))]
results=pd.DataFrame(results)
results=(results.T)
results.columns=["Train Result","Test Result"]
results.to_csv(".artifacts/result.csv",index=False)
