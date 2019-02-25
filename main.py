from networks.base_network import Generator, Discriminator, Classifier
from model import MSTN, fit 

from torch import optim

import loader.base_loader as loader

import argparse

import os

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--n_features', type=int, default=256, help='dimensionality of the featurespace')
parser.add_argument('--nc', type=int, default=256, help='dimensionality of the featurespace')

parser.add_argument('--center_interita', type=int, default=0.7, help='centers inertia over batches')


parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

parser.add_argument('--n_class', type=int, default=10, help='number of class')#to delete


parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')


args = parser.parse_args()




mstn = MSTN(args)

optim = optim.Adam(mstn.parameters(),lr = args.lr, betas= (args.b1, args.b2))# todo test with default settings


s_train, s_test = loader.mnist_loader(args)
t_train, t_test = loader.svhn_loader(args)


fit(args.epoch, mstn, optim, s_train,t_train, None, None)
