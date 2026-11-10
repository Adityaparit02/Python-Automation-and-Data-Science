from sklearn.datasets import load_iris

def main():
    print("-"*30)
    print("Iris Classification Case Study : ")
    print("-"*30)

    Dataset = load_iris()
    #Meta Data of DataSet

    print("Indepandant Variables are : ")
    print(Dataset.feature_names)

    print("Depandant Variables are : ")
    print(Dataset.target_names)
if __name__ == "__main__":
    main()