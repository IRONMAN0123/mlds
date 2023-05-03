# C:\Users\STUDENT\AppData\Local\Programs\Python\Python311\Scripts\pip install seaborn
#Write a program to implement k-Nearest Neighbor algorithm to classify the iris data set.
#Print both correct and wrong predictions



import sklearn
import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
#iris = pd.read_csv("iris.csv")
print(iris.keys())
df=pd.DataFrame(iris['data'])
print(df)
print(iris['target_names'])
print(iris['feature_names'])
print(iris['target'])

print("Feature Names")
print(iris.feature_names)

print("Target Names")
print(iris.target_names)

print("DataFrame with header Fields")
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())

print("shape and size of the dataset")
print(df.shape)

print("Index of the each colors with target")
df['target']=iris.target
print(df.head())
print(df[df.target==0].head())
print(df[df.target==1].head())
print(df[df.target==2].head())

print("Flower names with target of eacg features")
df['flower_name']=df.target.apply(lambda x: iris.target_names[x])
print(df.head())

print("instances with different indexes")
df0=df[:49]
df1=df[50:99]
df2=df[100:]

import matplotlib.pyplot as plt
print("sepal length and sepal width of setosa,versicolor, and virginica")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color="blue",marker='_')
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color="orange",marker='.')
plt.show()

print("petallength and petal width of setosa,versicolor, and virginica")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color="blue",marker='_')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color="orange",marker='.')
plt.show()

# Train and Test Split
X=df
y=iris['target']
X=iris.data
y=iris.target
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

print("Training data and Test data split")
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(X_test))
print(df)


#LOGISTICS REGRESSION
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X, y)
# predict the response values for the observations in X
logreg.predict(X)
y_pred = logreg.predict(X)
# check how many predictions were generated
len(y_pred)


# Create KNN
print("Create KNN")
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
knn = KNeighborsClassifier(n_neighbors=3)

print("KNN FIT",knn.fit(X, y))
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))

#from sklearn import metrics
#knn=KNeighborsClassifier(n_neighbors=3)
#print(knn)
#knn.fit(X,y)
#knn.fit(X_train,y_train)
print("knn score",knn.score(X_test,y_test))
print("Hi")

#Confusion Matrix
print("Confusion Matrix")
from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)



cm=confusion_matrix(y_test,y_pred)
print(cm)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True)
plt.xlabel=('Predicted')
plt.ylabel=('Truth')
plt.show()

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#Accuracy Score
from sklearn.metrics import accuracy_score
print("Correct prediction", accuracy_score(y_test,y_pred))
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
print("Wrong prediction", (1-accuracy_score(y_test,y_pred)))
y_testtrain=knn.predict(X_train)
cm1=confusion_matrix(y_train,y_testtrain)
print(cm1)






