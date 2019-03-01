import torch.nn as nn

from torchvision.models import AlexNet, alexnet

import torch.utils.model_zoo as model_zoo
from torch.autograd import Function
from torch import tensor

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}
 
class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self,args ):
        super(Generator, self).__init__()

        #self.alex = alexnet(pretrained= True)
        self.input_size=  args.input_size

        #self.alex.classifier = nn.Sequential(
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.avg = nn.AdaptiveAvgPool2d((6, 6))

        self.batch_norm = nn.BatchNorm2d(3)
        self.clf = nn.Sequential(
            
            #nn.Linear(64 * 6 * 6, 1024),

            nn.Linear(self.input_size, 1024),
            nn.ReLU(inplace= True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace= True),
            nn.Linear(1024, args.n_features),
        )
        

    def forward(self, x):
        x=self.batch_norm(x)
        #x= self.features(x)
        #x = self.avg(x)
        
        x = x.view(x.size(0),-1 )
        #x = x.view(x.size(0), 64 * 6 * 6)
        x = self.clf(x)
        return x


class AlexGen(AlexNet):
    """docstring for AlexGenerator"""
    def __init__(self, args):
        super(AlexGen, self).__init__()
        self.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, args.n_features),
        )

        for p in self.features.parameters():   
            p.requires_grad=False
    def forward(self, x):
        x = self.features(x).detach()
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x



class Discriminator(nn.Module):
    """docstring for Generator"""
    def __init__(self,args ):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(args.n_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.main(x)


class Classifier(nn.Module):
    """docstring for Generator"""
    def __init__(self,args):
        super(Classifier, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(args.n_features, args.n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.main(x)




class Rx(nn.Module):
    """docstring for Generator"""
   
    def forward(semf, i):
        return i

    def backward(semf, grad_output):
        return -grad_output