###################################################################################
#Step 1: Import required libraries
###################################################################################
border = "-"*70
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


###################################################################################
#Step 2: configuration of values
###################################################################################

VOCAB_SIZE = 10000              #consider most frequent 10000 unique words
MAX_LENGTH = 200                #consider maximum 200 words in review


###################################################################################
#Step 3: Load the imdb dataset
###################################################################################

print(border)
print("Movie Review Sentiment Analysis using LSTM")
print(border)


print("Loading the Dataset ...")

(X_train , Y_train) , (X_test , Y_test) = imdb.load_data(num_words = VOCAB_SIZE)

print("imdb Dataset Loaded Successfully..")

print("Number of Training Reviews : ",len(X_train))
print("Number of Testing Reviews : ",len(X_test))

###################################################################################
#   X_train        :   Reviews used for training 
#   X_test         :   Actual Sentiments of training
#   Y_train        :   Reviews used for testing
#   Y_test         :   Actual sentiment of testing

#   Sentiments:
#   0 -> Negative Sentiments
#   1 -> Positive Sentiments
###################################################################################

###################################################################################
#Step 4: Load the Word Dictionary
###################################################################################

word_index = imdb.get_word_index()

###################################################################################
#Dictionary Contains Mapping of word and its corresponding number
# Eg.
# Drishyam is good movie             -> (20  56  78  43)
# 20 -> Drishyam
# 56 -> is
# 78 -> good
# 43 -> movie
###################################################################################


###################################################################################
#Step 5: Create reverse Dictionary
###################################################################################

reverse_word_index = {}

for word,index in word_index.items():
    reverse_word_index[index+3] = word
