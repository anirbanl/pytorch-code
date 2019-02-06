import os
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from model import SNLIClassifier
from util import get_args, makedirs
import spacy
import json

args = get_args()
nlp = spacy.load('en')
SEED=1234
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

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=device)
else:
    model = SNLIClassifier(config)
    if args.word_vectors:
        model.embed.weight.data.copy_(inputs.vocab.vectors)
        model.to(device)

modelfile=os.path.join(args.save_path,[e for e in os.listdir(args.save_path) if '_'.join(e.split('_')[0:2])=='best_snapshot'][0])
print(modelfile)

model = torch.load(modelfile)
model.eval()

# calculate accuracy on test set
n_test_correct, test_loss = 0, 0
with torch.no_grad():
    for test_batch_idx, test_batch in enumerate(test_iter):
         answer = model(test_batch)
         n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()) == test_batch.label).sum().item()
         #test_loss = criterion(answer, test_batch.label)
test_acc = 100. * n_test_correct / len(test)

print('Test accuracy : %f'%(test_acc))

def predict_entailment(s1_premise,s2_hypothesis,label=''):
    tmap={}
    tmap['sentence1'],tmap['sentence2'],tmap['gold_label'] = s1_premise,s2_hypothesis,label
    with open('./.data/snli/snli_1.0/result.jsonl', 'w') as fp:
        json.dump(tmap, fp)
    a,b,c = datasets.SNLI.splits(inputs, answers, train='result.jsonl', validation='result.jsonl', test='result.jsonl')
    a_iter,b_iter,c_iter = data.BucketIterator.splits((a,b,c), batch_size=args.batch_size, device=device)
    batches=[(idx, batch) for idx, batch in enumerate(c_iter)]
    with torch.no_grad():
        answer=model(batches[0][1])
    return {1:'entailment',2:'contradiction',3:'neutral'}[torch.max(answer, 1)[1].item()]

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
