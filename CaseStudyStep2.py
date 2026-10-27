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
print("#Step 2 :   Data Analysis (EDA)")
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

