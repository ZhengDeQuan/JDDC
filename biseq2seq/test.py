import pickle
from biseq2seq import prepare_data
train_data = pickle.load(open('./data/train.pkl', 'rb'))
print("train")
vocab_hash = train_data.vocab_hash
for key,value in vocab_hash.items():
    if value == 20906:
        print(key,value)
        break

