# ht = tanh(Wx * Xt  + Wh * ht-1 +b)

#Xt         Current Input
#Wx         Weight of Current Input
#Wh         Weight of Previous hidden state
#b          Bias
#ht -1      Previous hidden state
#tanh       Activation function(-1 to 1)
#ht         New hidden state

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def MarvellousRNNPredictions():
    print("Calculations of RNN")


            #food was not good
    inputs = [1, 2, 5 , 3]

    hidden_state = 0


    #########################################################################
    # RNN Parameters
    #########################################################################

    Wx = 0.5    #weights for input
    Wh = 0.8    #weights for hidden state
    bias = 0.1

    for time_step , x in enumerate(inputs):
        previous_hidden_state = hidden_state

        weighted_input = Wx * x
        weighted_memory = Wh * previous_hidden_state


        total = weighted_input + weighted_memory + bias

        hidden_state = np.tanh(total)

        print("Time Step : ", time_step+1)
        print("Input : ",x)
        print("Hidden State : ",hidden_state)
        print("-" * 70)

    print("Final Hidden State : ",hidden_state)





def main():
    MarvellousRNNPredictions()


if __name__ == "__main__":
    main()