import torch
import spacy
import random
nlp = spacy.load('en')
SEED=1234
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchtext import data
TEXT = data.Field(lower=True, tokenize='spacy')
LABEL = data.Field(sequential=False)

from torchtext import datasets
train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, embed=True):

        if embed:
            #x = [sent len, batch size]
            x = x.permute(1, 0)
            #x = [batch size, sent len]
            embedded = self.embedding(x)
        else:
            #x = [sent len, batch size, emb_dim]
            x = x.permute(1, 0, 2)
            #x = [batch size, sent len, emb_dim]
            embedded = x

        #embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        #embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        #conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        #pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        #cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = len(LABEL.vocab)
DROPOUT = 0.5

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
model = model.to(device)

checkpoint=torch.load('./models/cnn_model_sst.tar')
model.load_state_dict(checkpoint['model_state_dict'])

def predict_sentiment(sentence, min_len=5):
    emb=embed_sentence(sentence)
    with torch.no_grad():
        model.eval()
        prediction = model(emb, embed=False)
    return LABEL.vocab.itos[torch.max(prediction, 1)[1].item()]

def embed_sentence(s, min_len=5):
    tokenized = [tok.text for tok in nlp.tokenizer(s.decode("utf-8"))]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    emb=model.embedding(tensor)
    return emb

print("This film is terrible",predict_sentiment("This film is terrible"))
print("This film is great",predict_sentiment("This film is great"))
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



