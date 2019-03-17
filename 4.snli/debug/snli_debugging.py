# coding: utf-8
x=torch.randn((50,10))
import torch
x=torch.randn((50,10))
class Simple(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Simple, self).__init__()
        self.lin=torch.nn.Linear(input_dim,hid_dim)
        self.lin=torch.nn.Linear(hid_dim,out_dim)
        
class Simple(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Simple, self).__init__()
        self.lin1=torch.nn.Linear(input_dim,hid_dim)
        self.lin2=torch.nn.Linear(hid_dim,out_dim)
    def forward(self, x):
        return self.lin2(self.lin1(x))
    
model=Simple(10,100,2)
y=torch.randint(2,(50,1))
y
criterion=nn.BCEWithLogitsLoss()
from torch import nn
criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters())
def train(model, optimizer, criterion):
    
    model.train()
    optimizer.zero_grad()
    logits=model(x)
    loss=criterion(logits,y.float())
    loss.backward()
    optimizer.step()
    return loss
def epoch_train(N=20):
    for epoch in range(N):
        l=train(model, optimizer, criterion)
        print(epoch,l)
        
epoch_train(2000)
def train(model, optimizer, criterion):
    
    model.train()
    optimizer.zero_grad()
    logits=torch.argmax(model(x),dim=1)
    loss=criterion(logits,y.float())
    loss.backward()
    optimizer.step()
    return loss
epoch_train(2000)
y=torch.randint(2,(50,))
epoch_train(2000)
def train(model, optimizer, criterion):
    
    model.train()
    optimizer.zero_grad()
    logits=torch.argmax(model(x),dim=1)
    loss=criterion(logits,y.float())
    loss.backward()
    optimizer.step()
    return loss
x.size()
logits=torch.argmax(model(x),dim=1)
logits.size()
logits
def train(model, optimizer, criterion):
    
    model.train()
    optimizer.zero_grad()
    logits=torch.argmax(model(x),dim=1).float()
    loss=criterion(logits,y.float())
    loss.backward()
    optimizer.step()
    return loss
def train(model, optimizer, criterion):
    
    model.train()
    optimizer.zero_grad()
    logits=model(x)
    loss=criterion(logits,y.float())
    loss.backward()
    optimizer.step()
    return loss
model=Simple(10,100,1)
epoch_train(2000)
y=torch.randint(2,(50,1))
epoch_train(2000)
x.requires_grad
t=torch.randn((1,10))
t
model.train()
model.zero_grad()
o=torch.sigmoid(model(t))
sg=torch.autograd.grad(o,t)
t.requires_grad=True
model.train()
model.zero_grad()
o=torch.sigmoid(model(t))
sg=torch.autograd.grad(o,t)
sg
sg[0]
x.requires_grad
class Simple(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Simple, self).__init__()
        self.lin1=torch.nn.Linear(input_dim,hid_dim)
        self.lin2=torch.nn.Linear(hid_dim,out_dim)
    def forward(self, x, y):
        i1=self.lin1(x)
        i2=self.lin1(y)
        return self.lin2(i1+i2)
    
    
y=torch.randn((50,10))
z=torch.randint(2,(50,1))
def train(model, optimizer, criterion):
    
    model.train()
    optimizer.zero_grad()
    logits=model(x,y)
    loss=criterion(logits,z.float())
    loss.backward()
    optimizer.step()
    return loss
model=Simple(10,100,1)
epoch_train(20)
t
u=torch.randn((1,10))
model.train()
model.zero_grad()
o=torch.sigmoid(model(t,u))
o
sg=torch.autograd.grad(o,t)
sg
sg=torch.autograd.grad(o,t)
t.requires_grad
u.requires_grad
u.requires_grad=True
sg=torch.autograd.grad(o,(t,u),retain_graph=True)
sg=torch.autograd.grad(o,t,retain_graph=True)
model.zero_grad()
model.train()
model.zero_grad()
o=torch.sigmoid(model(t,u))
sg=torch.autograd.grad(o,(t,u),retain_graph=True)
sg
model.train()
model.zero_grad()
o=torch.sigmoid(model(t,u))
sg1=torch.autograd.grad(o,t,retain_graph=True)
sg2=torch.autograd.grad(o,u,retain_graph=True)
sg1
sg2
t
u
class Simple(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Simple, self).__init__()
        self.lin1=torch.nn.Linear(input_dim,hid_dim)
        self.lin2=torch.nn.Linear(hid_dim,out_dim)
    def forward(self, x, y):
        i1=self.lin1(x)
        i2=self.lin1(y)
        return self.lin2(i1+i2)
    
    
x=torch.randn((4,50,10))
y=torch.randn((5,50,10))
o=model(x,y)
class Simple(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Simple, self).__init__()
        self.lin1=torch.nn.Linear(input_dim,hid_dim)
        self.lin2=torch.nn.Linear(hid_dim,out_dim)
    def forward(self, x, y):
        i1=self.lin1(x)
        i2=self.lin1(y)
        o1=self.lin2(i1)
        o2=self.lin2(i2)
        return torch.max(o1,dim=0)+torch.max(o2,dim=0)
    
    
    
model=Simple(10,100,1)
o=model(x,y)
o
o.size()
class Simple(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Simple, self).__init__()
        self.lin1=torch.nn.Linear(input_dim,hid_dim)
        self.lin2=torch.nn.Linear(hid_dim,out_dim)
    def forward(self, x, y):
        i1=self.lin1(x)
        i2=self.lin1(y)
        o1=self.lin2(i1)
        o2=self.lin2(i2)
        
        return torch.max(o1,dim=0)[0]+torch.max(o2,dim=0)[0]
    
    
    
o=model(x,y)
o
o[0]
o[1]
o[2]
o[3]
a=torch.randn((4,50,1))
b=torch.randn((5,50,1))
a.size()
b.size()
torch.max(a)
torch.max(a,dim=0)
torch.max(a,dim=0)[0]
torch.max(b,dim=0)[0]
torch.add(torch.max(a,dim=0)[0],torch.max(b,dim=0)[0])
class Simple(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Simple, self).__init__()
        self.lin1=torch.nn.Linear(input_dim,hid_dim)
        self.lin2=torch.nn.Linear(hid_dim,out_dim)
    def forward(self, x, y):
        i1=self.lin1(x)
        i2=self.lin1(y)
        o1=self.lin2(i1)
        o2=self.lin2(i2)
        
        return torch.add(torch.max(o1,dim=0)[0],torch.max(o2,dim=0)[0])
    
    
    
o=model(x,y)
o
class Simple(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Simple, self).__init__()
        self.lin1=torch.nn.Linear(input_dim,hid_dim)
        self.lin2=torch.nn.Linear(hid_dim,out_dim)
    def forward(self, x, y):
        i1=self.lin1(x)
        i2=self.lin1(y)
        o1=self.lin2(i1)
        o2=self.lin2(i2)
        return o1,o2
        #return torch.add(torch.max(o1,dim=0)[0],torch.max(o2,dim=0)[0])
    
    
    
o1,o2=model(x,y)
model=Simple(10,100,1)
class Simple(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Simple, self).__init__()
        self.lin1=torch.nn.Linear(input_dim,hid_dim)
        self.lin2=torch.nn.Linear(hid_dim,out_dim)
    def forward(self, x, y):
        i1=self.lin1(x)
        i2=self.lin1(y)
        o1=self.lin2(i1)
        o2=self.lin2(i2)
        #return o1,o2
        return torch.add(torch.max(o1,dim=0)[0],torch.max(o2,dim=0)[0])
    
    
    
model=Simple(10,100,1)
o=model(x,y)
o
o.size()
z.size()
def train(model, optimizer, criterion):
    
    model.train()
    optimizer.zero_grad()
    logits=model(x,y)
    loss=criterion(logits,z.float())
    loss.backward()
    optimizer.step()
    return loss
x.size()
y.size()
t=torch.randn((4,1,10))
u=torch.randn((5,1,10))
t.requires_grad
u.requires_grad
epoch_train(20)
model.train()
model.zero_grad()
o=model(t,u)
sg=torch.autograd.grad(o,(t,u),retain_graph=True)
t.requires_grad=True
u.requires_grad=True
sg=torch.autograd.grad(o,(t,u),retain_graph=True)
sg=torch.autograd.grad(o,t,retain_graph=True)
model.train()
model.zero_grad()
o=model(t,u)
sg=torch.autograd.grad(o,(t,u),retain_graph=True)
sg
sg[0].size()
sg[1].size()
sg[0].detach().numpy().sum(dim=2)
sum1=sg[0].sum(dim=2)
sum1
sum2=sg[1].sum(dim=2)
sum2
