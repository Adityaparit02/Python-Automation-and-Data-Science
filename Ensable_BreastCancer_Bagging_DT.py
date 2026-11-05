import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

############################################################################
#Step 1 : Load the Dataset
############################################################################

df = pd.read_csv("breast_cancer.csv")


print("Shape of Dataset : ", df.shape)
print("First Few Records")
print(df.head())


############################################################################
#Step 2 : Seperate Features and Labels
############################################################################

X = df.drop("target",axis=1)
Y = df["target"]

print("X Shape :",X.shape)
print("Y Shape :",Y.shape)


############################################################################
#Step 3 : Split Dataset for Training and Testing
############################################################################

X_train , X_test , Y_train ,Y_test = train_test_split(X,
                                                      Y,
                                                      test_size=0.2,
                                                      random_state=42) 


############################################################################
#Step 4 : Scale the Features
############################################################################

Scalar = StandardScaler()

X_train = Scalar.fit_transform(X_train)
X_test = Scalar.fit_transform(X_test)

############################################################################
#Step 5.1 : Create the Base Model
############################################################################

Base_Model = DecisionTreeClassifier(random_state=42)


############################################################################
#Step 5.2 : Create the Bagging Model
############################################################################

Model = BaggingClassifier(estimator=Base_Model,n_estimators=10,random_state=42)


############################################################################
#Step 6 : Train Model
############################################################################
Model = Model.fit(X_train,Y_train)


############################################################################
#Step 7 : Test the Model
############################################################################

Y_Pred = Model.predict(X_test)

############################################################################
#Step 8 : Evaluate the Model
############################################################################

print("-"*70)
accuracy = accuracy_score(Y_test,Y_Pred)
print("Accuracy of Model is : ",accuracy*100)
print("-"*70)
print("-"*70)

print("Confusion Matrix : ")
print(confusion_matrix(Y_test,Y_Pred))
print("-"*70)
