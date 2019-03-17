import torch
import torch.nn as nn
import os
from argparse import ArgumentParser

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def get_args():
    parser = ArgumentParser(description='PyTorch/torchinputs SNLI example')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_embed', type=int, default=100)
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.6B.100d')
    parser.add_argument('--resume_snap', type=str, default='')
    args = parser.parse_args()
    return args

import os
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import data
from torchtext import datasets
import random
import numpy as np

args = get_args()

seed=1234
def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['pythonhashseed'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs = data.Field(lower=True, tokenize='spacy')
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answers)

inputs.build_vocab(train, max_size=35000, vectors="glove.6B.100d")

# if args.word_vectors:
#     if os.path.isfile(args.vector_cache):
#         inputs.vocab.vectors = torch.load(args.vector_cache)
#     else:
#         inputs.vocab.load_vectors(args.word_vectors)
#         makedirs(os.path.dirname(args.vector_cache))
#         torch.save(inputs.vocab.vectors, args.vector_cache)
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=args.batch_size, device=device)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers

import torch.nn as nn
import torch.nn.functional as F

class Embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Embedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(RNN, self).__init__()
        
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        embedded = self.dropout(x)
        
        #embedded = [sent len, batch size, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        
        #output = [sent len, batch size, hid dim * num directions]
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
                
        #hidden = [batch size, hid dim * num directions]
            
        return self.fc(hidden.squeeze(0))

class SimpleSNLI(torch.nn.Module):
    def __init__(self, embedder, rnn, output_dim):
        super(SimpleSNLI, self).__init__()
        self.embedder=embedder
        self.rnn=rnn
        self.final=nn.Linear(2*output_dim, output_dim)
    def forward(self, p, h):
        pe, ph = self.embedder(p), self.embedder(h)
        oe, oh = self.rnn(pe), self.rnn(ph)
        return self.final(torch.cat((oe,oh), dim=1))


INPUT_DIM = len(inputs.vocab)
EMBEDDING_DIM = args.d_embed
HIDDEN_DIM = args.d_hidden
OUTPUT_DIM = len(answers.vocab)
N_LAYERS = args.n_layers
BIDIRECTIONAL = True
DROPOUT = 0.5

embedder = Embedder(INPUT_DIM, EMBEDDING_DIM)
rnn = RNN(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
classifier = SimpleSNLI(embedder, rnn, OUTPUT_DIM)

pretrained_embeddings = inputs.vocab.vectors

print(pretrained_embeddings.shape)

embedder.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

embedder = embedder.to(device)
rnn = rnn.to(device)
classifier = classifier.to(device)
criterion = criterion.to(device)

def accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # print(preds)
    # print(preds.size())
    # print(y)
    # print(y.size())
    n_correct = (torch.max(preds, 1)[1].view(y.size()) == y).sum().item()    
    n_total = preds.size()[0]
    acc = 1. * n_correct/n_total
    return acc

def train(classifier, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    classifier.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()

        predictions = classifier(batch.premise, batch.hypothesis)
        
        loss = criterion(predictions, batch.label)
        
        acc = accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(classifier, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    classifier.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = classifier(batch.premise, batch.hypothesis)
            
            loss = criterion(predictions, batch.label)
            
            acc = accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_train(N_EPOCHS = 10):
    best_valid_loss = 100.0

    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train(classifier, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(classifier, dev_iter, criterion)
        test_loss, test_acc = evaluate(classifier, test_iter, criterion)
        
        print('| Epoch: %d | Train Loss: %.3f | Train Acc: %.2f | Val. Loss: %.3f | Val. Acc: %.2f | Test Loss: %.3f | Test Acc: %.2f |'%(epoch+1,train_loss,train_acc*100,valid_loss,valid_acc*100,test_loss, test_acc*100))
        torch.save({
                'epoch': epoch+1,
                'rnn_state_dict': model.state_dict(),
                'emb_state_dict' : embedder.state_dict(),
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, './models/rnn_model_snli.tar')
        if valid_loss < best_valid_loss:
            torch.save({
                'epoch': epoch+1,
                'rnn_state_dict': model.state_dict(),
                'emb_state_dict' : embedder.state_dict(),
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, './models/best_rnn_classifier_snli.tar')
            best_valid_loss = valid_loss

    test_loss, test_acc = evaluate(classifier, test_iter, criterion)

    print('| Test Loss: %.3f | Test Acc: %.2f |'%(test_loss,test_acc*100))

epoch_train(20)


