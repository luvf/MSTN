import torch.nn as nn

from torchvision.models import alexnet


        
class Generator(nn.Module):
    """docstring for Generator"""
    def __init__(self,args ):
        super(Generator, self).__init__()

        self.alex = alexnet(pretrained= True)

        self.alex.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, args.n_features),
        )
        

    def forward(self, x):
        x = self.alex(x)
        return x
        #x = self.avgpool(x)
        #x = x.view(x.size(0), 256 * 6 * 6)#flatten
        #out = self.main(x)
        #return out




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
            nn.Linear(256, args.n_class),
            nn.Softmax()
        )
            


    def forward(self, x):
        return self.main(x)

