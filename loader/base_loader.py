


from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def mnist_loader(args):
	transform = transforms.Compose([])
	train = datasets.MNIST("dataset/mnist", train = True, download= True, transform = transform)
	test  = datasets.MNIST("dataset/mnist", train = False, download= True, transform= transform )

	train_loader = DataLoader(train, batch_size= args.batch_size)
	test_loader = DataLoader(test, batch_size= args.batch_size)
	return train_loader, test_loader



def svhn_loader(args):
	transform = transforms.Compose([])
	train = datasets.SVHN("dataset/svhn", split = "train", download= True, transform = transform)
	test  = datasets.SVHN("dataset/svhn", split = "test", download= True, transform= transform )

	train_loader = DataLoader(train, batch_size= args.batch_size)
	test_loader = DataLoader(test, batch_size= args.batch_size)
	return train_loader, test_loader


"""

class TransferLoader:
    def __init__(self, source, target):
        self.source = spirce
        self.target = target
        
        #self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)

        for b in batches:
            yield (self.func(*b))
            """