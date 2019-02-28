import os

import argparse
import torch


from networks.base_network import AlexGen
from networks.model import MSTN, fit, MSTNoptim

import loader.base_loader as loader


os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--n_features', type=int, default=256, help='dimensionality of the featurespace')
parser.add_argument('--nc', type=int, default=256, help='dimensionality of the featurespace')



parser.add_argument('--lr', type=float, default=0.005, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.8, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.8, help='adam: decay of first order momentum of gradient')

parser.add_argument('--center_interita', type=int, default=0.7, help='centers inertia over batches')
#parser.add_argument('--adv_w', type=float, default=0.999, help='adversarial wheight')
#parser.add_argument('--lam', type=float, default=0., help='semantic wheight')

parser.add_argument('--n_class', type=int, default=10, help='number of class')#to delete


parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')

parser.add_argument('--save', type=str, default="trained/model", help='dir of the trained_model')
parser.add_argument('--save_step', type=int, default=0, help='dir of the trained_model')

parser.add_argument('--load', type=str, default=None, help='dir of the trained_model')
parser.add_argument('--set_device', type=str, default="cpu", help='set cuda')

parser.add_argument('--dataset', type=str, default="chiffres", help='choosing dataset')

parser.add_argument('--input_size', type=int, default=28*28*3, help='choosing dataset')

args = parser.parse_args()
args.device = None
if args.set_device == "cuda" and torch.cuda.is_available():
	args.device = torch.device('cuda')
	print("cuda enabled")
else:
	args.device = torch.device('cpu')

#mstn = MSTN(args,gen= AlexGen(args)).to(device = args.device)


mstn = MSTN(args).to(device = args.device)


if args.load != None:
    mstn.load_state_dict(torch.load(args.load))


optim = torch.optim.Adam(mstn.parameters(), lr = args.lr, betas= (args.b1, args.b2), weight_decay = 0.005)
#optim = torch.optim.RMSprop(mstn.parameters(), lr = args.lr, momentum= 0.9, weight_decay = 0.005)
if (args.dataset == 'chiffres'):
	s_train, s_test = loader.mnist_loader(args)
	t_train, t_test = loader.svhn_loader(args)
	trainset = loader.TransferLoader(s_train,t_train)
	teststet = loader.TransferLoader(s_test,t_test)
elif (args.dataset == 'office_31'):
	s_train, s_test = loader.amazon_loader(args)
	t_train, t_test = loader.webcam_loader(args)
	trainset = loader.TransferLoader(s_train,t_train)
	teststet = loader.TransferLoader(s_test,t_test)
else:
	s_train, s_test = loader.amazon_loader(args)
	t_train, t_test = loader.webcam_loader(args)
	trainset = loader.TransferLoader(s_train,t_train)
	teststet = loader.TransferLoader(s_test,t_test)



fit(args, args.epoch, mstn, optim, trainset, teststet)


torch.save(mstn.state_dict(), args.save)
