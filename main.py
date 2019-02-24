from networks.base_network import Generator, Discriminator, Classifier
from model import MSTN
from torch import optim

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')

parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--center_interita', type=int, default=0.7, help='centers inertia over batches')

args = parser.parse_args()




mstn = MSTN(args)

optim = optim.Adam(model.parameters(),lr = args.lr, betas= (args.b1, args.b2))# todo test with default settings

fit(args.epoch, mstn, optim, train_dl, None, None)
