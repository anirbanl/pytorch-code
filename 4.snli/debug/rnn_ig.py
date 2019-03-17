import torch
import torch.nn as nn


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        dropout = 0 if config.n_layers == 1 else config.dp_ratio
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=dropout,
                        bidirectional=config.birnn)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 =  inputs.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class Embedder(nn.Module):
    def __init__(self, config):
        super(Embedder, self).__init__()
        self.config = config
        self.embedding_dim = config.d_embed
        self.embedding = nn.Embedding(config.n_embed, config.d_embed)

    def forward(self, x):
        return self.embedding(x)

class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super(SNLIClassifier, self).__init__()
        self.config = config
        # self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        seq_in_size = 2*config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2
        lin_config = [seq_in_size]*2
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
            Linear(seq_in_size, config.d_out))

    def forward(self, x, sizes):
        prem_embed, hypo_embed = torch.split(x, sizes)
        if self.config.fix_emb:
            prem_embed =prem_embed.detach()
            hypo_embed =hypo_embed.detach()
        if self.config.projection:
            prem_embed = self.relu(self.projection(prem_embed))
            hypo_embed = self.relu(self.projection(hypo_embed))
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)
        scores = self.out(torch.cat([premise, hypothesis], 1))
        return scores

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
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    parser.add_argument('--epochs', type=int, default=20)
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
seed_everything(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inputs = data.Field(lower=args.lower, tokenize='spacy')
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answers)

inputs.build_vocab(train, dev, test)

if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        inputs.vocab.vectors = torch.load(args.vector_cache)
    else:
        inputs.vocab.load_vectors(args.word_vectors)
        makedirs(os.path.dirname(args.vector_cache))
        torch.save(inputs.vocab.vectors, args.vector_cache)
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=args.batch_size, device=device)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers

# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2

if args.resume_snap:
    model = torch.load(args.resume_snap, map_location=device)
else:
    embedder = Embedder(config)
    model = SNLIClassifier(config)
    if args.word_vectors:
        embedder.embedding.weight.data.copy_(inputs.vocab.vectors)
        embedder.to(device)
        model.to(device)

criterion = nn.CrossEntropyLoss()
opt = O.Adam(model.parameters(), lr=args.lr)

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
        model.train(); opt.zero_grad()

        iterations += 1
        p_emb, h_emb = embedder(batch.premise), embedder(batch.hypothesis)
        emb=torch.cat([p_emb,h_emb],dim=0)

        # forward pass
        answer = model(emb, [p_emb.size()[0], h_emb.size()[0]])

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
            torch.save(model, snap_path)
            for f in glob.glob(snap_prefix + '*'):
                if f != snap_path:
                    os.remove(f)

        # evaluate performance on validation set periodically
        if iterations % args.dev_every == 0:

            # switch model to evaluation mode
            model.eval(); dev_iter.init_epoch()

            # calculate accuracy on validation set
            n_dev_correct, dev_loss = 0, 0
            with torch.no_grad():
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                     p_emb, h_emb = embedder(dev_batch.premise), embedder(dev_batch.hypothesis)
                     emb=torch.cat([p_emb,h_emb],dim=0)
                     answer = model(emb, [p_emb.size()[0], h_emb.size()[0]])
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
                torch.save(model, snap_path)
                for f in glob.glob(snap_prefix + '*'):
                    if f != snap_path:
                        os.remove(f)

        elif iterations % args.log_every == 0:

            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), ' '*8, n_correct/n_total*100, ' '*12))

import spacy
import json

nlp = spacy.load('en')
model.eval()

# calculate accuracy on test set
n_test_correct, test_loss = 0, 0
with torch.no_grad():
    for test_batch_idx, test_batch in enumerate(test_iter):
         p_emb, h_emb = embedder(test_batch.premise), embedder(test_batch.hypothesis)
         emb=torch.cat([p_emb,h_emb],dim=0)
         answer = model(emb, [p_emb.size()[0], h_emb.size()[0]])
         n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()) == test_batch.label).sum().item()
         #test_loss = criterion(answer, test_batch.label)
test_acc = 100. * n_test_correct / len(test)

print('Test accuracy : %f'%(test_acc))

def predict_entailment(s1_premise,s2_hypothesis,label=''):
    p_emb, h_emb = embed_pair(s1_premise, s2_hypothesis,label)
    with torch.no_grad():
        model.eval()
        answer = model(emb, [p_emb.size()[0], h_emb.size()[0]])
    return answers.vocab.itos[torch.max(answer, 1)[1].item()]

def embed_pair(s1_premise, s2_hypothesis, label):
    tmap={}
    tmap['sentence1'],tmap['sentence2'],tmap['gold_label'] = s1_premise,s2_hypothesis,label
    with open('./.data/snli/snli_1.0/result.jsonl', 'w') as fp:
        json.dump(tmap, fp)
    a,b,c = datasets.SNLI.splits(inputs, answers, train='result.jsonl', validation='result.jsonl', test='result.jsonl')
    a_iter,b_iter,c_iter = data.BucketIterator.splits((a,b,c), batch_size=args.batch_size, device=device)
    batches=[(idx, batch) for idx, batch in enumerate(c_iter)]
    emb_p, emb_h = embedder(batches[0][1].premise), embedder(batches[0][1].hypothesis)
    return emb_p, emb_h

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
    p_emb, h_emb = embed_pair(s1_premise,s2_hypothesis,'')
    x=torch.cat([p_emb,h_emb],dim=0)
    x_dash = torch.zeros_like(x)
    sum_grad = None
    grad_array = None
    x_array = None
    with torch.no_grad():
        model.eval()
        pred=torch.argmax(model(x,[p_emb.size()[0], h_emb.size()[0]]))
    model.train()
    for k in range(m):
        model.zero_grad()
        step_input = x_dash + k * (x - x_dash) / m
        step_output = model(step_input,[p_emb.size()[0], h_emb.size()[0]])
        step_pred = torch.argmax(step_output)
        step_grad = torch.autograd.grad(step_output[0,pred], x, retain_graph=True)[0]
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
    return relevances

integrated_gradients("A black race car starts up in front of a crowd of people.","A man is driving down a lonely road.")

