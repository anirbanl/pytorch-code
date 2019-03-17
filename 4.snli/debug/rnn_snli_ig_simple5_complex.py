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

args = get_args()

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
#optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# encoder1 = encoder1.to(device)
# encoder2 = encoder2.to(device)
embp = embp.to(device)
embh = embh.to(device)
classifier = classifier.to(device)
criterion = criterion.to(device)
emb = (embp, embh)

#criterion = nn.CrossEntropyLoss()
opt = optim.Adam(classifier.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
makedirs(args.save_path)
print(header)

for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):

        # switch model to training mode, clear gradient accumulators
        classifier.train(); opt.zero_grad()

        iterations += 1
        p, h = embp(batch.premise), embh(batch.hypothesis)

        # forward pass
        answer = classifier(p, h)

        # calculate accuracy of predictions in the current batch
        n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels
        loss = criterion(answer, batch.label)

        # backpropagate and update optimizer learning rate
        loss.backward(); opt.step()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            snap_prefix = os.path.join(args.save_path, 'snap')
            snap_path = snap_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.item(), iterations)
            torch.save(classifier, snap_path)
            for f in glob.glob(snap_prefix + '*'):
                if f != snap_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:

            # switch model to evaluation mode
            classifier.eval(); dev_iter.init_epoch()

            # calculate accuracy on validation set
            n_dev_correct, dev_loss = 0, 0
            with torch.no_grad():
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                     p, h = embp(dev_batch.premise), embh(dev_batch.hypothesis)
                     answer = classifier(p, h)
                     n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()) == dev_batch.label).sum().item()
                     dev_loss = criterion(answer, dev_batch.label)
            dev_acc = 100. * n_dev_correct / len(dev)

            print(dev_log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))

            # update best valiation set accuracy
            if dev_acc > best_dev_acc:

                # found a model with better validation set accuracy

                best_dev_acc = dev_acc
                snap_prefix = os.path.join(args.save_path, 'best_snap')
                snap_path = snap_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)

                # save model, delete previous 'best_snap' files
                torch.save(classifier, snap_path)
                for f in glob.glob(snap_prefix + '*'):
                    if f != snap_path:
                        os.remove(f)

        elif iterations % args.log_every == 0:

            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), ' '*8, n_correct/n_total*100, ' '*12))

# calculate accuracy on test set
n_test_correct, test_loss = 0, 0
with torch.no_grad():
    for test_batch_idx, test_batch in enumerate(test_iter):
         p, h = embp(test_batch.premise), embh(test_batch.hypothesis)
         answer = classifier(p, h)
         n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()) == test_batch.label).sum().item()
         #test_loss = criterion(answer, test_batch.label)
test_acc = 100. * n_test_correct / len(test)

print('Test accuracy : %f'%(test_acc))

epoch_train(5)

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

