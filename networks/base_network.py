import torch.nn as nn

from torchvision.models import alexnet

class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self,args ):

        self.alex = alexnet(pretrained= True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.main = nn.Sequential(
            
            #linpretrained
            nn.Linear(6*6*256, 4096),
            nn.ReLu(inplace=True),
            nn.Dropout(0.5),#parametrize

            nn.Linear(4096, 4096),
            nn.ReLu(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, args.n_features),
        )


    def forward(self, x):
        x = self.alex(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)#flatten
        out = self.main(x)


        return out




class Discriminator(nn.Module):
    """docstring for Generator"""
    def __init__(self,args ):
        
        self.main = nn.Sequential(
            nn.Linear(args.n_features, 1024),
            nn.ReLu(),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 1024),
            nn.ReLu(),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.main(x)


class Classifier(nn.Module):
    """docstring for Generator"""
    def __init__(self,args):
        
        self.main = nn.Sequential(
            nn.Linear(256, args.n_class),
            nn.softmax()
        )
            


    def forward(self, x):
        return self.main(x)

