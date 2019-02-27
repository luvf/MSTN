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

office_31 = {'amazon': 'office_31/amazon/images/',
			'dslr': 'office_31/dslr/images/',
			'webcam': 'office_31/webcam/images/'}

office_transform = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

def amazon_loader(args):
	amazon_data = datasets.ImageFolder(
			office_31['amazon'],
			transform=office_transform
		)
	train_size = int(0.8 * len(amazon_data))
	test_size = len(amazon_data) - train_size
	train, test = torch.utils.data.random_split(amazon_data, [train_size, test_size])

	train_loader = DataLoader(train, batch_size= args.batch_size, shuffle=True)
	test_loader = DataLoader(test, batch_size= args.batch_size, shuffle=True)
	return train_loader, test_loader
  
def webcam_loader(args):
	webcam_data = datasets.ImageFolder(
			office_31['webcam'],
			transform=office_transform
		)
	train_size = int(0.8 * len(webcam_data))
	test_size = len(webcam_data) - train_size
	train, test = torch.utils.data.random_split(webcam_data, [train_size, test_size])

	train_loader = DataLoader(train, batch_size= args.batch_size, shuffle=True)
	test_loader = DataLoader(test, batch_size= args.batch_size, shuffle=True)
	return train_loader, test_loader
  
def dslr_loader(args):
	dslr_data = datasets.ImageFolder(
			office_31['dslr'],
			transform=office_transform
		)
	train_size = int(0.8 * len(dslr_data))
	test_size = len(dslr_data) - train_size
	train, test = torch.utils.data.random_split(dslr_data, [train_size, test_size])

	train_loader = DataLoader(train, batch_size= args.batch_size, shuffle=True)
	test_loader = DataLoader(test, batch_size= args.batch_size, shuffle=True)
	return train_loader, test_loader



class TransferLoader:
	def __init__(self, source, target):
		self.source = source
		self.target = target
		
		#self.func = func

	def __len__(self):
		return min(2, len(self.source), len(self.target))

	def __iter__(self):
		s = iter(self.source)
		t = iter(self.target)
		for _ in range(len(self)-1):
			yield (*s.next(), *t.next())
