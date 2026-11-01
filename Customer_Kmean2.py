import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import k_means


def main():
    ############################################################################
    #Step 1: Load Data
    ############################################################################

    df = pd.read_csv("Mall_Customers.csv")

    print("Data set loaded with values")
    print(df.head())
    
    print("Missing Values")
    print(df.isnull().sum())

    ############################################################################
    #Step 2: Feature Selection
    ############################################################################

    X= df[["AnnualIncome","SpendingScore"]]
    print("Selected Features")
    print(X.head())


    ############################################################################
    #Step 3: Scale the Data
    ############################################################################
    scalar = StandardScaler()

    X_Scaled = scalar.fit_transform(X)

    print("Scaled Data")
    print(X_Scaled[:10])

if __name__ == "__main__":
    main()