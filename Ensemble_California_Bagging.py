import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score,mean_squared_error


border = "=" * 70
############################################################################
#Step 1 : Load the Dataset
############################################################################
df = pd.read_csv("california_housing.csv")

print(border)
print("Shape of Dataset:")
print(df.shape)


print(border)
print("First few records :")
print(df.head())


############################################################################
#Step 2 : Seperate Features and Labels
############################################################################
X = df.drop("target",axis=1)
Y = df["target"]

print(border)
print("Shape of X : ")
print(X.shape)
print("-"*70)
print("Shape of Y: ")
print(Y.shape)

############################################################################
#Step 3 : Split Dataset for Training and Testing
############################################################################

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


############################################################################
#Step 4.1 : Create the Base Model
############################################################################

Base_Model = DecisionTreeRegressor(random_state=42)

############################################################################
#Step 4.2 : Create the Bagging Model
############################################################################

Model = BaggingRegressor(estimator=Base_Model, n_estimators=10, random_state=42)


############################################################################
#Step 5 : Train the Model
############################################################################

Model = Model.fit(X_train,Y_train)

############################################################################
#Step 6 : Test the Model
############################################################################

Y_Pred = Model.predict(X_test)

############################################################################
#Step 7 : Evaluate the model
############################################################################


print(border)
MSE = mean_squared_error(Y_test,Y_Pred)
print("MSE : ",MSE)

R2 = r2_score(Y_test,Y_Pred)
print(border)
print("R2 Score is : ",R2)

print(border)
