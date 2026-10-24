from sklearn import tree

def main():
    print("Ball Classification Case Study")

    Independant = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0]]
    #Testing Features = [35,1],[95,0]


    Dependant = [1,1,2,1,2,1,2,1,1,1,2,1,2]
    #Testing Labels = [1,2]
    
    print("Features are :",Independant)
    print("Labels are :",Dependant)

if __name__ == "__main__":
    main()


    