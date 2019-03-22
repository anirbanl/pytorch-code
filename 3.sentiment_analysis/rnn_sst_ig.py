import torch
import random
import numpy as np
import os
import spacy
nlp = spacy.load('en')
SEED=1234
def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)

from torchtext import data
TEXT = data.Field(lower=True, tokenize='spacy')
LABEL = data.Field(sequential=False)

from torchtext import datasets
train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE, 
    device=device)

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

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = len(LABEL.vocab)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

embedder = Embedder(INPUT_DIM, EMBEDDING_DIM)
model = RNN(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

embedder.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

embedder = embedder.to(device)
model = model.to(device)
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

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()

        embeddings = embedder(batch.text)
        
        predictions = model(embeddings)
        
        loss = criterion(predictions, batch.label)
        
        acc = accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            embeddings = embedder(batch.text)

            predictions = model(embeddings)
            
            loss = criterion(predictions, batch.label)
            
            acc = accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_train(N_EPOCHS = 10):
    best_valid_loss = 100.0

    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
        test_loss, test_acc = evaluate(model, test_iterator, criterion)
        
        print('| Epoch: %d | Train Loss: %.3f | Train Acc: %.2f | Val. Loss: %.3f | Val. Acc: %.2f | Test Loss: %.3f | Test Acc: %.2f |'%(epoch+1,train_loss,train_acc*100,valid_loss,valid_acc*100,test_loss, test_acc*100))
        torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'emb_state_dict' : embedder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, './models/rnn_model_sst.tar')
        if valid_loss < best_valid_loss:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'emb_state_dict' : embedder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, './models/best_rnn_model_sst.tar')
            best_valid_loss = valid_loss

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print('| Test Loss: %.3f | Test Acc: %.2f |'%(test_loss,test_acc*100))

epoch_train(20)

def embed_sentence(s):
    tokenized = [tok.text for tok in nlp.tokenizer(s.decode("utf-8"))]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    return embedder(tensor)

def predict_sentiment(s):
    emb=embed_sentence(s)
    with torch.no_grad():
        model.eval()
        prediction = model(emb).unsqueeze(0)
    return LABEL.vocab.itos[torch.max(prediction, 1)[1].item()]

print("This film is terrible",predict_sentiment("This film is terrible"))
print("This film is great",predict_sentiment("This film is great"))
print("this film is great",predict_sentiment("this film is great"))
print("this film is good",predict_sentiment("this film is good"))
print("this film is bad",predict_sentiment("this film is bad"))
print("This film is not bad",predict_sentiment("This film is not bad"))
print("My friend likes awesome food",predict_sentiment("My friend likes awesome food"))
print("My friend likes awful recipes",predict_sentiment("My friend likes awful recipes"))
print("the film is amazingly delightful to watch",predict_sentiment("the film is amazingly delightful to watch"))
print("the film is boring",predict_sentiment("the film is boring"))
print("the film is not good",predict_sentiment("the film is not good"))
print("the film is fun",predict_sentiment("the film is fun"))
print("the film is awful",predict_sentiment("the film is awful"))
print("the film is bad",predict_sentiment("the film is bad"))
print("the film is a true story",predict_sentiment("the film is a true story"))
print("the film is a fake story",predict_sentiment("the film is a fake story"))
print("i like this phone",predict_sentiment("i like this phone"))
print("i hate this phone",predict_sentiment("i hate this phone"))
print("the camera is priceless",predict_sentiment("the camera is priceless"))
print("the camera is expensive",predict_sentiment("the camera is expensive"))

import pandas as pd

def integrated_gradients(s, m=300):
    x=embed_sentence(s)
    x_dash = torch.zeros_like(x)
    sum_grad = None
    grad_array = None
    x_array = None
    with torch.no_grad():
        model.eval()
        pred=torch.argmax(model(x))
    model.train()
    for k in range(m):
        model.zero_grad()
        step_input = x_dash + k * (x - x_dash) / m
        step_output = model(step_input)
        step_pred = torch.argmax(step_output)
        step_grad = torch.autograd.grad(step_output[pred], x)[0]
        if sum_grad is None:
            sum_grad = step_grad
            grad_array = step_grad
            x_array = step_input
        else:
            sum_grad += step_grad
            grad_array = torch.cat([grad_array, step_grad])
            x_array = torch.cat([x_array, step_input])
    sum_grad = sum_grad / m
    sum_grad = sum_grad * (x - x_dash)
    sum_grad = sum_grad.sum(dim=2)
    tokens=s.split(' ')
    relevances = sum_grad.detach().cpu().numpy()
    #print(list(np.round(np.reshape(relevances,len(tokens)),3)))
    try:
        relevances = list(np.round(np.reshape(relevances,len(tokens)),3))
        df = pd.DataFrame(index=['Sentence','IntegGrad'], columns=list(range(len(tokens))), data=[tokens, relevances])
        print("Sentence : %s"%(s))
        with pd.option_context('display.max_rows', None, 'display.max_columns', 30):
            print(df)
        print("PREDICTED Label : %s"%(LABEL.vocab.itos[pred]))
        return LABEL.vocab.itos[pred], relevances
    except:
        print "*****Error*******"
        return LABEL.vocab.itos[pred], []

integrated_gradients("This film is terrible")
integrated_gradients("This film is great")
integrated_gradients("this film is great")
integrated_gradients("this film is good")
integrated_gradients("this film is bad")
integrated_gradients("This film is not bad")
integrated_gradients("My friend likes awesome food")
integrated_gradients("My friend likes awful recipes")
integrated_gradients("the film is amazingly delightful to watch")
integrated_gradients("the film is boring")
integrated_gradients("the film is not good")
integrated_gradients("the film is fun")
integrated_gradients("the film is awful")
integrated_gradients("the film is bad")
integrated_gradients("the film is a true story")
integrated_gradients("the film is a fake story")
integrated_gradients("i like this phone")
integrated_gradients("i hate this phone")
integrated_gradients('the camera is priceless')
integrated_gradients('the camera is expensive')

igmap={}
count=0
for i in range(len(test_data)):
    print(''.join(200*['-']))
    p, r =integrated_gradients(' '.join(test_data.examples[i].__dict__['text']).encode('utf-8'))
    l=test_data.examples[i].__dict__['label']
    print("TRUE Label : %s"%(l))
    match= (p==l)
    print("%s"%({0:"WRONG",1:"CORRECT"}[int(match)]))
    print(''.join(200*['-']))
    if match:
        count+=1
    igmap[' '.join(test_data.examples[i].__dict__['text'])]=(p,r)
print("Test accuracy : %f"%(count * 100.0 / len(test_data)))

import cPickle as cp
with open('rnn_sst_ig.pkl','wb') as fp:
    cp.dump(igmap, fp)

