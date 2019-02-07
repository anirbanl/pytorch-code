import torch
import spacy
import random
nlp = spacy.load('en')
SEED=1234
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchtext import data
TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)

from torchtext import datasets

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

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

    def forward(self, x):

        #x = [sent len, batch size]

        x = x.permute(1, 0)

        #x = [batch size, sent len]

        embedded = self.embedding(x)

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
OUTPUT_DIM = 1
DROPOUT = 0.5

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
model = model.to(device)

checkpoint=torch.load('./models/cnn_model.tar')
model.load_state_dict(checkpoint['model_state_dict'])

def predict_sentiment(sentence, min_len=5):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence.decode("utf-8"))]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    with torch.no_grad():
        model.eval()
        prediction = torch.sigmoid(model(tensor))
    return prediction.item()

print("This film is terrible",predict_sentiment("This film is terrible"))
print("This film is great",predict_sentiment("This film is great"))
print("This film is not bad",predict_sentiment("This film is not bad"))
print("My friend likes awesome food",predict_sentiment("My friend likes awesome food"))
print("My friend likes awful recipes",predict_sentiment("My friend likes awful recipes"))
