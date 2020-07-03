"""hw4_RNN"""

"""Specify Path"""
import sys
test_fname    = sys.argv[1]
model_fname   = sys.argv[2]
predict_fname = sys.argv[3]
w2v_fname     = 'w2v_all.model'

"""Utils"""
def load_testing_data(path=test_fname):
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X

"""Data Preprocess"""
from torch import nn
from gensim.models import Word2Vec

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path=w2v_fname):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    
    def get_w2v_model(self):
        # read in word to vec model
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
    
    def add_embedding(self, word):
        # add word into embedding, and give it a random representation vector
        # word can only be "<PAD>" or "<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    
    def make_embedding(self, load=True):
        print("Get embedding ...")
        # get trained Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # construct word2idx  dictionary
        # construct idx2word  list
        # construct word2vector  list
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i+1), end='\r')
            # e.g. self.word2index['he'] = 1 
            # e.g. self.index2word[1] = 'he'
            # e.g. self.vectors[1] = 'he' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # add "<PAD>" and "<UNK>" into embedding
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    
    def pad_sequence(self, sentence):
        # every sentense has the same length
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    
    def sentence_word2idx(self):
        # from word to corresponding index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # every sentense has the same length
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    
    def labels_to_tensor(self, y):
        # turn labels into tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)

"""Dataset"""
import torch
from torch.utils import data

class TwitterDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)
    
    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return len(self.data)

"""Model"""
import torch
from torch import nn
class BiLSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(BiLSTM_Net, self).__init__()
        # construct embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # fix_embedding = False -> embedding will be trained
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim*2, 1),
                                         nn.Sigmoid() )
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x

"""Testing"""
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1 # >= 0.5 negative
            outputs[outputs <  0.5] = 0 # <  0.5 positive
            ret_output += outputs.int().tolist()
    
    return ret_output

"""Main - Predict and Write to csv file"""
import torch
import pandas as pd

sen_len = 30
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# test model and predict
print("loading testing data ...")
test_x = load_testing_data(test_filename)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_fname)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
print('\nload model ...')
model = torch.load(model_fname)
outputs = testing(batch_size, test_loader, model, device)

# write to csv 
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
print("save csv ...")
tmp.to_csv((predict_fname), index=False)
print("Finish Predicting")