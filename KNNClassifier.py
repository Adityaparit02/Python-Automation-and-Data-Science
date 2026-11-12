import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def main():
    #Independant
    X = np.array([
        [1,2],
        [2,3],
        [3,1],
        [5,6]
    ])

    #Dependant
    Y = np.array(["Red","Red","Blue","Blue"])

    new_point = np.array([[3,3]])

    #Model Creation
    Model = KNeighborsClassifier(n_neighbors=3)
    Model = Model.fit(X,Y)

    Y_Pred = Model.predict(new_point)

    print("Predicted Label : ", Y_Pred)
    

if __name__=="__main__":
    main()