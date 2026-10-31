import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import k_means


def main():
    #Step 1:


    df = pd.read_csv("Mall_Customers.csv")

    print("Data set loaded with values")
    print(df.head())
    
    print("Missing Values")
    print(df.isnull().sum())

    

if __name__ == "__main__":
    main()