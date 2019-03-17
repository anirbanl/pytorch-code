import torch
import torch.nn as nn
import os
from argparse import ArgumentParser
import time
import glob
import torch.optim as O
import torch.nn.functional as F

from torchtext import data
from torchtext import datasets
import random
import numpy as np

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
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=128, device=device)

class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass

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
            
        return self.fc(hidden)

class SimpleSNLI(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(SimpleSNLI, self).__init__()
        self.projection = Linear(embedding_dim, embedding_dim)
        self.encoder=RNN(embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        lin_config = [output_dim*2]*2
        self.out = nn.Sequential(
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(*lin_config),
            self.relu,
            self.dropout,
            Linear(output_dim*2, output_dim))
    def forward(self, p, h):
        ip = self.relu(self.projection(p))
        ih = self.relu(self.projection(h))
        op, oh = self.encoder(ip), self.encoder(ih)
        return self.out(torch.cat((op, oh), dim=1))

INPUT_DIM = len(inputs.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 300
OUTPUT_DIM = len(answers.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2

# encoder1 = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
# encoder2 = Encoder(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
embp=Embedder(INPUT_DIM, EMBEDDING_DIM)
embh=Embedder(INPUT_DIM, EMBEDDING_DIM)
classifier = SimpleSNLI(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
INPUT_DIM

pretrained_embeddings = inputs.vocab.vectors

print(pretrained_embeddings.shape)

# encoder1.embedder.embedding.weight.data.copy_(pretrained_embeddings)
# encoder2.embedder.embedding.weight.data.copy_(pretrained_embeddings)
embp.embedding.weight.data.copy_(pretrained_embeddings)
embh.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# encoder1 = encoder1.to(device)
# encoder2 = encoder2.to(device)
embp = embp.to(device)
embh = embh.to(device)
classifier = classifier.to(device)
criterion = criterion.to(device)
emb = (embp, embh)

def accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    n_correct = (torch.max(preds, 1)[1].view(y.size()) == y).sum().item()    
    n_total = preds.size()[0]
    acc = 1. * n_correct/n_total
    return acc

def train(emb, classifier, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    emb[0].train()
    emb[1].train()
    classifier.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        ep, eh = emb[0](batch.premise), emb[1](batch.hypothesis)

        predictions = classifier(ep, eh)
        
        loss = criterion(predictions, batch.label)
        
        acc = accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(emb, classifier, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    emb[0].eval()
    emb[1].eval()    
    classifier.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
        
            ep, eh = emb[0](batch.premise), emb[1](batch.hypothesis)

            predictions = classifier(ep, eh)
            
            loss = criterion(predictions, batch.label)
            
            acc = accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_train(N_EPOCHS = 10):
    best_valid_loss = 100.0

    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train(emb, classifier, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(emb, classifier, dev_iter, criterion)
        test_loss, test_acc = evaluate(emb, classifier, test_iter, criterion)
        
        print('| Epoch: %d | Train Loss: %.3f | Train Acc: %.2f | Val. Loss: %.3f | Val. Acc: %.2f | Test Loss: %.3f | Test Acc: %.2f |'%(epoch+1,train_loss,train_acc*100,valid_loss,valid_acc*100,test_loss, test_acc*100))

    test_loss, test_acc = evaluate(emb, classifier, test_iter, criterion)

    print('| Test Loss: %.3f | Test Acc: %.2f |'%(test_loss,test_acc*100))

epoch_train(20)

import json
def predict_entailment(s1_premise,s2_hypothesis,label=''):
    p_emb, h_emb = embed_pair(s1_premise, s2_hypothesis,label)
    with torch.no_grad():
        classifier.eval()
        answer = classifier(p_emb, h_emb)
    return answers.vocab.itos[torch.max(answer, 1)[1].item()]

def embed_pair(s1_premise, s2_hypothesis, label):
    tmap={}
    tmap['sentence1'],tmap['sentence2'],tmap['gold_label'] = s1_premise,s2_hypothesis,label
    with open('./.data/snli/snli_1.0/result.jsonl', 'w') as fp:
        json.dump(tmap, fp)
    a,b,c = datasets.SNLI.splits(inputs, answers, train='result.jsonl', validation='result.jsonl', test='result.jsonl')
    a_iter,b_iter,c_iter = data.BucketIterator.splits((a,b,c), batch_size=128, device=device)
    batches=[(idx, batch) for idx, batch in enumerate(c_iter)]
    embp.eval()
    embh.eval()
    p_emb, h_emb = embp(batches[0][1].premise), embp(batches[0][1].hypothesis)
    return p_emb, h_emb

print(predict_entailment("A black race car starts up in front of a crowd of people.","A man is driving down a lonely road."))
print(predict_entailment("A soccer game with multiple males playing.","Some men are playing a sport."))
print(predict_entailment("A smiling costumed woman is holding an umbrella.","A happy woman in a fairy costume holds an umbrella."))
print(predict_entailment("A person on a horse jumps over a broken down airplane.","A person is training his horse for a competition."))
print(predict_entailment("A person on a horse jumps over a broken down airplane.","A person is at a diner, ordering an omelette."))
print(predict_entailment("A person on a horse jumps over a broken down airplane.","A person is outdoors, on a horse."))
print(predict_entailment("A person on a horse jumps over a broken down airplane.","A person is indoors, on a horse."))
print(predict_entailment("A person on a horse jumps over a broken down airplane.","A person is outside, on a horse."))
print(predict_entailment("A person on a horse jumps over a sofa.","A person is outside, on a horse."))
print(predict_entailment("A person is beside a horse.","A person is outside, on a horse."))
print(predict_entailment("A person is beside a boy.","A person is outside, on a horse."))

def integrated_gradients(s1_premise, s2_hypothesis, m=300):
    p, h = embed_pair(s1_premise,s2_hypothesis,'')
    p_dash, h_dash = torch.zeros_like(p), torch.zeros_like(h)
    sum_grad = None
    grad_array = None
    x_array = None
    with torch.no_grad():
        classifier.eval()
        pred=torch.argmax(classifier(p, h))
    classifier.train()
    for k in range(m):
        classifier.zero_grad()
        step_input_p, step_input_h = p_dash + k * (p - p_dash) / m, h_dash + k * (h - h_dash) / m
        step_output = classifier(step_input_p, step_input_h)
        step_pred = torch.argmax(step_output)
        step_grad = torch.autograd.grad(step_output[0,pred], (p, h), retain_graph=True)
        if sum_grad is None:
            sum_grad = [step_grad[0], step_grad[1]]
#             grad_array = (step_grad[0], step_grad[1])
#             p_array, h_array = step_input_p, step_input_h
        else:
            sum_grad[0] += step_grad[0]
            sum_grad[1] += step_grad[1]
#             grad_array = torch.cat([grad_array, step_grad])
#             p_array, h_array = torch.cat([p_array, step_input_p]), torch.cat([h_array, step_input_h])
    sum_grad[0], sum_grad[1] = sum_grad[0] / m, sum_grad[1] / m
    sum_grad[0], sum_grad[1] = sum_grad[0] * (p - p_dash), sum_grad[1] * (h - h_dash)
    sum_grad[0], sum_grad[1] = sum_grad[0].sum(dim=2), sum_grad[1].sum(dim=2)
    relevances = (sum_grad[0].detach().cpu().numpy(), sum_grad[1].detach().cpu().numpy())
    return relevances

integrated_gradients("A black race car starts up in front of a crowd of people.","A man is driving down a lonely road.")

