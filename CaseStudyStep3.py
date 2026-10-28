import pandas as pd
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


