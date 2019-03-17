# coding: utf-8
get_ipython().magic(u'save simple1 1-21')
class Simple(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Simple, self).__init__()
        self.linx1=torch.nn.Linear(input_dim,hid_dim)
        self.liny1=torch.nn.Linear(input_dim,hid_dim)
        self.lin2=torch.nn.Linear(hid_dim,out_dim)
    def forward(self, x, y):
        i1=self.linx1(x)
        i2=self.liny1(y)
        o1=self.lin2(i1)
        o2=self.lin2(i2)
        #return o1,o2
        return torch.add(torch.max(o1,dim=0)[0],torch.max(o2,dim=0)[0])
model=Simple(10,100,1)
x=torch.randn((4,50,10))
y=torch.randn((5,50,10))
def train(model, optimizer, criterion):

    model.train()
    optimizer.zero_grad()
    logits=model(x,y)
    loss=criterion(logits,z.float())
    loss.backward()
    optimizer.step()
    return loss
from torch import nn
criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters())
def epoch_train(N=20):
    for epoch in range(N):
        l=train(model, optimizer, criterion)
        print(epoch,l)
model=Simple(10,100,1)
x=torch.randn((4,50,10))
y=torch.randn((5,50,10))
epoch_train(20)
t=torch.randn((4,1,10))
u=torch.randn((5,1,10))
t.requires_grad
u.requires_grad
t.requires_grad=True
u.requires_grad=True
model.train()
model.zero_grad()
o=model(t,u)
o
t.requires_grad
u.requires_grad
sg=torch.autograd.grad(o,(t,u),retain_graph=True)
sg
