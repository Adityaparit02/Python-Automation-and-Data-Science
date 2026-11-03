from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def main():
    iris_Data =  load_iris()

    X = iris_Data.data
    Y = iris_Data.target

    X_Train , x_Test , Y_Train , Y_Test = train_test_split(X,Y,test_size=0.5,random_state=42)

    Model = DecisionTreeClassifier()

    Model = Model.fit(X_Train,Y_Train)

    Y_Pred = Model.predict(x_Test)

    Result = accuracy_score(Y_Test,Y_Pred)

    print("Accuracy is : ",Result*100)

if __name__ == "__main__":
    main()