import torch
import torch.nn as nn
from torch import Tensor

from torch import optim

from networks.base_network import Generator, Discriminator, Classifier


import numpy as np


from tqdm import tqdm



class MSTNoptim():
    """docstring for optim algo"""
    def __init__(self, model, args):
        super(MSTNoptim, self).__init__()
        base_params = list(model.gen.parameters()) +  list(model.clf.parameters())
        self.base = optim.Adam(base_params, lr = args.lr, betas= (args.b1, args.b2))
        self.dis = optim.Adam(model.dis.parameters(),lr = args.lr, betas= (args.b1, args.b2))#




class MSTN(nn.Module):
    """docstring for MSTN algo"""
    def __init__(self, args, gen= None, dis = None, clf = None):
        super(MSTN, self).__init__()
            
        if gen == None :
            self.gen = Generator(args)
        if dis == None :
            self.dis = Discriminator(args)
        if clf == None :
            self.clf = Classifier(args)

        #self.train = True

        #not the cleanest way
        self.n_features = args.n_features
        self.n_class = args.n_class
        self.s_center = torch.zeros((args.n_class, args.n_features), requires_grad = False, device=args.device)
        self.t_center = torch.zeros((args.n_class, args.n_features), requires_grad = False, device=args.device)
        self.disc = args.center_interita

    #def train_model(self, train = True):
    #    self.dtrain = train

    def forward(self, x):
        features = self.gen(x)

        C_out = self.clf(features)
        #if self.train :
        D_out = self.dis(features)
        return C_out, features, D_out
        #else :
        #    return C_out




def update_centers(model, s_gen, t_gen, s_true, t_clf, args):
    source = torch.argmax(s_true, 1).reshape(t_clf.size(0),1)# one Hot 
    target = torch.argmax(t_clf, 1).reshape(t_clf.size(0),1)

    s_center = torch.zeros(model.n_class, model.n_features, device=args.device)
    t_center = torch.zeros(model.n_class, model.n_features, device=args.device)

    s_zeros = torch.zeros(source.size()[1:], device=args.device)
    t_zeros = torch.zeros(target.size()[1:], device=args.device)

    for i in range(model.n_class):
        s_cur = torch.where(source.eq(i), s_gen, s_zeros).mean(0)
        t_cur = torch.where(target.eq(i), t_gen, t_zeros).mean(0)
        s_center[i] = s_cur * (1 - model.disc) + model.s_center[i] * model.disc
        t_center[i] = t_cur * (1 - model.disc) + model.t_center[i] * model.disc

    
    return s_center, t_center
    #return s_class, t_class

adversarial_loss = torch.nn.BCELoss()
classification_loss =  torch.nn.MSELoss()

def loss_batch(model, sx, tx, s_true, opt, args):
    opt.base.zero_grad()
    opt.dis.zero_grad()
    
    s_clf, s_gen, s_dis = model(sx)
    t_clf, t_gen, t_dis = model(tx)
    #helpers
    source_tag = torch.ones((sx.size(0), 1), device = args.device)
    target_tag = torch.zeros((tx.size(0), 1), device = args.device)
    s_true_hot = one_hot(s_true, model.n_class).to(device=args.device)
    #classification loss
    C_loss = classification_loss(s_clf, s_true_hot)

    #generator loss
    s_G_loss = adversarial_loss(s_dis, target_tag)#0
    t_G_loss = adversarial_loss(t_dis, source_tag)#1
    G_loss = (s_G_loss + t_G_loss)/2

   
    #center loss more tricky
    s_c, t_c = update_centers(model, s_gen, t_gen, s_true_hot, t_clf, args)
    
    
    model.s_center = s_c.detach()
    model.t_center = t_c.detach()

    S_loss = classification_loss(t_c, s_c)

    loss = S_loss + C_loss + G_loss
   
    loss.backward()
    
    # Discrimanator loss
    opt.dis.zero_grad()

    s_dis = model.dis(s_gen.detach())
    t_dis = model.dis(t_gen.detach())

    #s_dis = s_dis.to(device=args.device)
    #t_dis = t_dis.to(device=args.device)

    s_D_loss = adversarial_loss(s_dis, source_tag)#0
    t_D_loss = adversarial_loss(t_dis, target_tag)#1

    D_loss = (s_D_loss + t_D_loss)/2

    D_loss.backward()
    opt.dis.step()
    return S_loss.item(), C_loss.item(), G_loss.item(), D_loss.item()



def eval_batch(model, sx, tx, s_true, t_true,args):
    s_clf, s_gen, s_dis = model(sx)
    t_clf, t_gen, t_dis = model(tx)

    #helpers
    source_tag = torch.ones((sx.size(0), 1), device = args.device)
    target_tag = torch.zeros((tx.size(0), 1), device = args.device)
    s_true_hot = one_hot(s_true, model.n_class).to(device=args.device)

    #classification loss
    c_loss = classification_loss(s_clf, s_true_hot)

    #generator loss
    s_G_loss = adversarial_loss(s_dis, target_tag)#0
    t_G_loss = adversarial_loss(t_dis, source_tag )#1
    G_loss = (s_G_loss + t_G_loss)/2
    #center loss more tricky
    s_c, t_c = update_centers(model, s_gen, t_gen, s_true_hot, t_clf,args)
    model.s_center = s_c.detach()
    model.t_center = t_c.detach()
    s_loss = classification_loss(t_c, s_c)

    #adversarial loss
    s_D_loss = adversarial_loss(s_dis, source_tag)#0
    t_D_loss = adversarial_loss(t_dis, target_tag)#1

    D_loss = (s_D_loss + t_D_loss)/2

    acc = metric(t_clf, t_true, torch.nn.MSELoss(),args)

    return np.array([acc, s_loss.item(), c_loss.item(), G_loss.item(), D_loss.item()])

    
def fit(args, epochs, model, opt, dataset, valid):
    out = list()
    for epoch in range(epochs):
        model.train()

        for sx, sy, tx,_ in tqdm(dataset):
            loss = loss_batch(model, sx.to(device= args.device), tx.to(device= args.device), sy, opt, args)
        model.eval()
        with torch.no_grad():
            loss = np.zeros(5)
            for sx, sy, tx, ty in tqdm(valid):
                loss += eval_batch(model, sx.to(device= args.device), tx.to(device= args.device), sy, ty,args)
            loss /= len(valid)-1
            print(epoch, loss)
        out.append((epoch, loss))
        if args.save_step:
            torch.save(model.state_dict(), args.save+'step')
    return out

#utils


def one_hot(batch,classes):
    ones = torch.eye(classes)
    return ones.index_select(0,batch)

from itertools import permutations

def metric_help(pred, true, loss,args):
    return min([loss(pred,one_hot(torch.tensor([perm[i] for i in true]),args.n_class).to(device = args.device)) for perm in permutations(range(args.n_class)) ] )


def metric(pred,true, loss, args):
    n_class = args.n_class

    true = true.reshape(pred.size(0),1)
    zeros = torch.zeros((pred.size(0), 1), device = args.device)
    i_class = torch.zeros_like(true)
    for i in range(n_class):
        sum_class = torch.where(true.eq(i), pred, zeros).sum(0)
        i_class[i]= torch.argmax(sum_class)

    true2 = one_hot(torch.tensor([i_class[i] for i in true]),n_class)
    return loss(pred, true2)