import torch
import spacy
import random
nlp = spacy.load('en')
SEED=1234
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

from torchtext import data
TEXT = data.Field(tokenize='spacy', preprocessing=generate_bigrams)
LABEL = data.LabelField(dtype=torch.float)

from torchtext import datasets

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(FastText, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x, embed=True):

        if embed:
            #x = [sent len, batch size]
            embedded = self.embedding(x)
        else:
            embedded = x

        #embedded = [sent len, batch size, emb dim]

        embedded = embedded.permute(1, 0, 2)

        #embedded = [batch size, sent len, emb dim]

        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        #pooled = [batch size, embedding_dim]

        return self.fc(pooled)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1

model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM)
model = model.to(device)

checkpoint=torch.load('./models/fasttext_model.tar')
model.load_state_dict(checkpoint['model_state_dict'])

def predict_sentiment(sentence):
    emb=embed_sentence(sentence)
    with torch.no_grad():
        model.eval()
        prediction = torch.sigmoid(model(emb,embed=False))
    return prediction.item()

def embed_sentence(s):
    tokenized = [tok.text for tok in nlp.tokenizer(s.decode("utf-8"))]
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
