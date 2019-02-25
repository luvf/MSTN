import torch
import torch.nn as nn
from torch import Tensor


from networks.base_network import Generator, Discriminator, Classifier


import numpy as np








class MSTN(nn.Module):
    """docstring for MSTN"""
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

        self.n_class = args.n_class
        self.s_center = torch.zeros(args.n_class, requires_grad = False)
        self.t_center = torch.zeros(args.n_class, requires_grad = False)
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




def update_centers(model, s_gen, t_gen, s_true, t_clf):
    source = torch.argmax(s_true, 1).reshape(t_clf.size(0),1)# one Hot 
    target = torch.argmax(t_clf, 1).reshape(t_clf.size(0),1)

    #shape = [t_clf.size(1)]+list(t_gen.size())
    #s_class =  torch.zeros(shape)
    #t_class =  torch.zeros(shape)

    s_zeros = torch.zeros(source.size()[1:])
    t_zeros = torch.zeros(target.size()[1:])

    for i in range(model.n_class):
        s_cur = torch.where(source.eq(i), s_gen, s_zeros)
        t_cur = torch.where(target.eq(i), t_gen, t_zeros)
        
        model.s_center[i]= s_cur.mean() *(1-model.disc) + model.s_center[i] * model.disc
        model.t_center[i]= t_cur.mean() *(1-model.disc) + model.t_center[i] * model.disc

    #return s_class, t_class

adversarial_loss = torch.nn.BCELoss()
classification_loss =  torch.nn.MSELoss()

def one_hot(batch,depth):
    ones = torch.eye(depth)
    return ones.index_select(0,batch)


def loss_batch(model, sx, tx, s_true, opt=None):
    s_clf, s_gen, s_dis = model(sx)
    t_clf, t_gen, t_dis = model(tx)

    s_true_hot = one_hot(s_true, model.n_class)
   
    #classification loss
    c_loss = classification_loss(s_clf, s_true_hot)

    # adversarial loss
    source_tag = Tensor(sx.size(0), 1).fill_(1.0)
    target_tag = Tensor(tx.size(0), 1).fill_(0.0)
    source_loss = adversarial_loss(s_gen, source_tag)#0
    target_loss = adversarial_loss(t_gen, target_tag)#1
    d_loss = source_loss + target_loss
   
    #center loss more tricky
    update_centers(model, s_gen, t_gen, s_true_hot, t_clf)
    s_loss = classification_loss(model.s_center, model.t_center)

    loss = s_loss  + c_loss + d_loss

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()








def fit(epochs, model, opt, dataset, eval_func, valid_dl):
    for epoch in range(epochs):
            model.train()
            for sx, sy, tx,_ in dataset:
                loss = loss_batch(model, sx, tx, sy, opt)
            print(epoch,loss)

            #TODO
            #model.eval()
            #with torch.no_grad():
            #    losses, nums = zip(
            #        *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            #    )
            #val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            #print(epoch, val_loss)



