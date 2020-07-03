"""hw4_RNN"""

"""Specify Path"""
import sys

train_label_fname    = sys.argv[1]
train_no_label_fname = sys.argv[2]
model_fname          = sys.argv[3]
w2v_fname            = 'w2v_all.model'

"""Utils"""
def load_training_data(path=train_label_fname):
    # read in training data
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x

def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs >= 0.5] = 1 # >= 0.5 negative
    outputs[outputs <  0.5] = 0 # < 0.5 positive
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

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

"""Train"""
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

def training(batch_size, n_epoch, lr, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    
    model.train() # optimizer can update parameters
    criterion = nn.BCELoss() # define loss function : binary cross entropy loss
    t_batch = len(train) 
    v_batch = len(valid) 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc, best_acc = 0, 0, 0
    
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long) # device = "cuda",  inputs -> torch.cuda.LongTensor
            labels = labels.to(device, dtype=torch.float) # device = "cuda", labels -> torch.cuda.FloatTensor
            
            optimizer.zero_grad() # because loss.backward() will accumulate gradient, gradient should be zero-lized for every batch
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels) # calc training loss
            loss.backward() # calc loss's gradient
            
            optimizer.step() # update parameters
            correct = evaluation(outputs, labels) # calc training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        # validation
        model.eval() # model's parameters would be fixed
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)  # device = "cuda", inputs -> torch.cuda.LongTensor
                labels = labels.to(device, dtype=torch.float) # device = "cuda", labels -> torch.cuda.FloatTensor
                outputs = model(inputs)
                outputs = outputs.squeeze()
                
                loss = criterion(outputs, labels) # calc validation loss
                correct = evaluation(outputs, labels) # calc validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save(model, model_fname                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train()

"""Main - Training"""
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters definition
sen_len       = 30
fix_embedding = True # fix embedding during training
batch_size    = 128
epoch         = 10
lr            = 0.001

print("loading training data ...") # read in 'training_label.txt' and 'training_nolabel.txt'
train_x, y = load_training_data(train_label_fname)
train_x_no_label = load_training_data(train_no_label_fname)

# preprocess input and labels
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_fname)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

# construct model
model = BiLSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=2, dropout=0.8, fix_embedding=fix_embedding)
model = model.to(device)

# split data into training data and validation data
X_train, X_val, y_train, y_val = train_x[:180000], train_x[180000:], y[:180000], y[180000:]

# turn data into dataset for dataloader
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset   = TwitterDataset(X=X_val, y=y_val)

# turn data into batch of tensors
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)

# training
training(batch_size, epoch, lr, train_loader, val_loader, model, device)