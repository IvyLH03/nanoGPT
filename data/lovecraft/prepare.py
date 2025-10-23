"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import shutil
import requests
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# get the stoi and itos from shakespeare_char dataset
shakespeare_meta_path = os.path.join(os.path.dirname(__file__), '..', 'shakespeare_char', 'meta.pkl')
with open(shakespeare_meta_path, 'rb') as f:
    shakespeare_meta = pickle.load(f)
shakespeare_stoi = shakespeare_meta['stoi']
shakespeare_itos = shakespeare_meta['itos']

# create a mapping from characters to integers
# for any character not in shakespeare_itos, create a new index
stoi = shakespeare_meta['stoi']
itos = shakespeare_meta['itos']
next_index = 0
for ch in chars:
    if ch not in shakespeare_stoi:
        stoi[ch] = next_index + len(shakespeare_stoi)
        itos[next_index + len(shakespeare_stoi)] = ch
        next_index += 1

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# make splits for different lengths
data_size = 10000
while(data_size < len(data)):
    tmp_data = data[:data_size]
    train_data = tmp_data[:int(data_size*0.9)]
    val_data = tmp_data[int(data_size*0.9):]
    train_ids = encode(train_data)
    val_ids = encode(val_data)

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    # create a folder for the current data size
    os.makedirs(os.path.join(os.path.dirname(__file__), f'{data_size}'), exist_ok=True)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), f'{data_size}/train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), f'{data_size}/val.bin'))

    # copy meta.pkl
    shutil.copy2(os.path.join(os.path.dirname(__file__), 'meta.pkl'), os.path.join(os.path.dirname(__file__), f'{data_size}/meta.pkl'))

    data_size += 10000