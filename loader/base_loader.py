

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

tt= transforms.Lambda(lambda x: print(x.size()))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


mnist_trannsform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        normalize
    ])

#def to_onehot(n_class):
#    return transforms.Compose(
#    [
#        transforms.ToTensor(),
#        transforms.Lambda(lambda x: torch.zeros(1,n_class).scatter(1,x.reshape(1,1),1))
#    ])


def mnist_loader(args):
    
    train = datasets.MNIST("dataset/mnist", train = True, download= True, transform = mnist_trannsform)#, target_transform = to_onehot(args.n_class))
    test  = datasets.MNIST("dataset/mnist", train = False, download= True, transform= mnist_trannsform)#, target_transform = to_onehot(args.n_class))

    train_loader = DataLoader(train, batch_size= args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size= args.batch_size, shuffle=True)
    return train_loader, test_loader


svhn_trannsform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ])



def svhn_loader(args):
    train = datasets.SVHN("dataset/svhn", split = "train", download= True, transform = svhn_trannsform)#, target_transform = to_onehot(args.n_class))
    test  = datasets.SVHN("dataset/svhn", split = "test" , download= True, transform = svhn_trannsform)#, target_transform = to_onehot(args.n_class))

    train_loader = DataLoader(train, batch_size= args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size= args.batch_size, shuffle=True)
    return train_loader, test_loader



class TransferLoader:
    def __init__(self, source, target):
        self.source = source
        self.target = target
        
        #self.func = func

    def __len__(self):
        return min(40,len(self.source), len(self.target))

    def __iter__(self):
        s = iter(self.source)
        t = iter(self.target)
        for _ in range(len(self)):
            yield (*s.next(), *t.next())

