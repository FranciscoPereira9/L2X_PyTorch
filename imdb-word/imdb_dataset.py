import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from keras.preprocessing import sequence
from keras.datasets import imdb
try:
   import cPickle as pkl
except:
   import pickle as pkl

# Reproducibility:
torch.manual_seed(10086)
torch.cuda.manual_seed(1)
np.random.seed(10086)
random.seed(10086)
# Set parameters:
max_features = 5000
maxlen = 400

class IMDB_SentimentDataset(Dataset):

    def __init__(self, data_path="data", setting="train", split_val=0.05, model="original"):
        if 'id_to_word.pkl' not in os.listdir(data_path):
            # Load data from original dataset
            (x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=max_features, index_from=3)
            word_to_id = imdb.get_word_index()
            word_to_id = {k: (v + 3) for k, v in word_to_id.items()}
            word_to_id["<PAD>"] = 0
            word_to_id["<START>"] = 1
            word_to_id["<UNK>"] = 2
            id_to_word = {value: key for key, value in word_to_id.items()}
            # Pad Reviews
            x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
            x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
            y_train = np.eye(2)[y_train]
            y_val = np.eye(2)[y_val]
            # Save Data
            np.save(data_path+'/x_train.npy', x_train)
            np.save(data_path+'/y_train.npy', y_train)
            np.save(data_path+'/x_val.npy', x_val)
            np.save(data_path+'/y_val.npy', y_val)
            with open(data_path+'/id_to_word.pkl', 'wb') as f:
                pkl.dump(id_to_word, f)
        assert (setting == "train" or setting == "val" or setting == "test"), \
            "Parameter <setting> must be 'train', 'val' or 'test'."
        if setting == "train":
            self.data = np.load(data_path + '/x_train.npy')
            self.targets = np.load(data_path + '/y_train.npy')
            split = int(len(self.data)*(1-split_val))
            self.data = self.data[:split]
            self.targets = self.targets[:split]
        elif setting == "val":
            self.data = np.load(data_path + '/x_train.npy')
            self.targets = np.load(data_path + '/y_train.npy')
            split = int(len(self.data) * (1-split_val))
            self.data = self.data[split:]
            self.targets = self.targets[split:]
        else:
            self.data = np.load(data_path + '/x_val.npy')
            self.targets = np.load(data_path + '/y_val.npy')

        if model=="l2x":
            if setting=="train":
                assert 'pred_train.npy' in os.listdir(data_path),\
                    "File 'pred_train.npy' is not in the 'data' folder. Generate prediction first. "
                self.targets = np.load('data/pred_train.npy')
            if setting == "test":
                assert 'pred_test.npy' in os.listdir(data_path), \
                    "File 'pred_train.npy' is not in the 'data' folder. Generate prediction first. "
                self.targets = np.load('data/pred_test.npy')

        with open(data_path + '/id_to_word.pkl', 'rb') as f:
            self.id_to_word = pkl.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        target = self.targets[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float)
        }

    def get_text(self, idx):
        sentence = []
        for num in self.data[idx]:
            if num == 0:
                sentence.append("-")
            else:
                sentence.append(self.id_to_word[num])
        return " ".join(sentence)

if __name__=="__main__":
    data = IMDB_SentimentDataset(data_path="data", setting="train")
    data_val = IMDB_SentimentDataset(data_path="data", setting="val")
    data_test = IMDB_SentimentDataset(data_path="data", setting="test")
    print(data[0])