from transformers import AutoTokenizer

###############################################################################
#   Load PreTrained Tokenizer
###############################################################################

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Input Sentence
sentence = " I love AI"

# Convert the sentence into tokens
tokens = tokenizer.tokenize(sentence)

token_ids  = tokenizer.encode(
    sentence,
    add_special_tokens = False
)


# Convert token IDs back to text
decoded = tokenizer.decode(token_ids)

print("Original Sentence : ", sentence)
print("Tokens : ", tokens)
print("Token IDs : ",token_ids)
print("Decode Sentences : ",decoded)