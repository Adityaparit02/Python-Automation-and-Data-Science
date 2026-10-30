import pandas as pd         #EDA

import matplotlib.pyplot as plt         #Visualization
import seaborn as sns

from  sklearn.model_selection import train_test_split       #spilliting data in train and test

from sklearn.tree import DecisionTreeClassifier             #model

Border = "-" *30


##########################################################################################################
#Step 1 : Load The Dataset
##########################################################################################################
print(Border)
print("Step 1 : Load the Dataset")
print(Border)

DataPath = "iris.csv"
df = pd.read_csv(DataPath)

print("Dataset Loaded Successfully...")
print("Initial Entries from dataset are : ")
print(df.head)



##########################################################################################################
#Step 2 :   Data Analysis (EDA)
##########################################################################################################
print(Border)
print("Step 2 :   Data Analysis (EDA)")
print(Border)

print("Shape of Dataset : ",df.shape)
print()
print("Column Names: ",df.columns)
print()

print("Missing Values per column : ")
print(df.isnull().sum())
print()

print("Class distribution (Species count)")
print(df["species"].value_counts())
print()

print("Statistical Report of Dataset : ")
print(df.describe())




##########################################################################################################
#Step 3 :   Decide Independant and Dependant variables
##########################################################################################################
print(Border)
print("Step 3 :   Decide Independant and Dependant variables")
print(Border)


# x : Independant variable / Features
# y : Dependant variable / Labels

feture_cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
    ]

X = df[feture_cols]
Y = df["species"] 

print(" X shape : ",X.shape)
print(" Y shape : ",Y.shape)



##########################################################################################################
#Step 4 :   Visualisation of Dataset
##########################################################################################################
print(Border)
print("Step 4 :   Visualisation of Dataset")
print(Border)

#Scatter Plot
plt.figure(figsize=(7,5))
for sp in df["species"].unique():
    temp = df[df["species"] == sp]
    plt.scatter(temp["petal length (cm)"] , temp["petal width (cm)"],label = sp)

plt.title("Marvellous Iris Case Study")

plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")

plt.legend()
plt.grid()
plt.show()



##########################################################################################################
#Step 5 :   Split Dataset for Training and Testing 
##########################################################################################################
print(Border)
print("Step 5 :   Split Dataset for Training and Testing ")
print(Border)

X_Train , X_Test , Y_Train ,Y_Test = train_test_split(X,Y,test_size=0.5,random_state=42)

print("Dataset splitting activity completed")

print("X : ",X.shape)       #(150,4)
print("Y : ",Y.shape)       #(150,)

print("X_Train : ",X_Train.shape)   #(75,4)
print("X_Test: ",X_Test.shape)      #(75.4)

print("Y_Train : ",Y_Train.shape)   #(75,)
print("Y_Test : ",Y_Test.shape)     #(75,)

##########################################################################################################
#Step 6 :   Build the Model 
##########################################################################################################
print(Border)
print("Step 6 :   Build the Model  ")
print(Border)

model = DecisionTreeClassifier(max_depth=5)
print("Model gets Created")


##########################################################################################################
#Step 7 :   Train the Model 
##########################################################################################################
print(Border)
print("Step 7 :   Train the Model ")
print(Border)

model.fit(X_Train,Y_Train)

print("Model Trained Successfully")


##########################################################################################################
#Step 8 :   Test the Model 
##########################################################################################################
print(Border)
print("Step 8 :   Test the Model  ")
print(Border)

Y_pred = model.predict(X_Test)

print("Model Testing Done ")

print("Expected Answers : ")
print(Y_Test)

print("Predicted Answers :")
print(Y_pred)