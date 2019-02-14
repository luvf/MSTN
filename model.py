from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras import losses
import numpy as np
from networks import base_network


import keras.backend as K

class MSTN(object):
    """docstring for MSTN"""
    def __init__(self, input_dim, nb_class, Netwok  ):
        super(MSTN, self).__init__()
        self.nb_class = nb_class
        self.gamma = 0.3
        self.batch_size = 1
        self.sem_loss= 0


        self.latent_space_dim = (200,200,3)
        self.network = Netwok
        self.model = self.create_model()

        self.CenterS = np.zeros([nb_class]+list(self.latent_space_dim))
        self.CenterT = np.zeros([nb_class]+list(self.latent_space_dim))



    def create_model(self):
        
        losses = {
            
            "dis": self.D_loss,#lambda x,y  : self.D_loss(x,y),
            "gen": self.S_loss,#lambda x,y  : self.S_loss(x,y), 
            "clf": self.C_loss,#lambda x,y  : self.C_loss(x,y)
            }


        losses_wheights = {
           
            'dis' : 1,
            'clf' : 1,
            'gen' : 1,

        }

        
        inputs = Input(shape=(300,300,3))
        gen = self.network.gen_generator(inputs)#self.latent_space_dim)

        dis = self.network.gen_discriminor(gen)
        
        clf = self.network.gen_classifier(gen,self.nb_class)




        model = Model(inputs, name='MSTN', outputs = [ dis,gen, clf])
        model.summary()

        model.compile(loss=losses,loss_weights=losses_wheights, optimizer="adam")
        return model
  
  


 


    def D_loss(self,y_true,y_pred):
        return losses.binary_crossentropy(y_true,y_pred)

    def C_loss(self, y_true, y_pred):
        
        def map(x):
            return x != -1 
        #Ignore the square error if y_true[i] is near zero
        sgn = K.map_fn(map,y_true)

        return K.mean(sgn * K.square(y_true-y_pred),axis=-1)
        #for i, y in enumerate(y_true):
        #    if y != [-1]:
        #        pr.append(y_pred[i])
        #        tr.append(y_true[i])
        #return losses.categorical_crossentropy(tr,pr)

        
            

    def S_loss(self, y_true,y_pred):
        return K.variable(self.sem_loss / self.batch_size)+ K.zeros_like(y_pred)

    def gen_batch(Source, Target,batch_size):
        DataS = [(x, 0, 1, y)   for x, y in Source]
        DataT = [(x, 0, 0 ,[-1]) for x, y in Target]
        Data = DataS+DataT
        np.random.shuffle(Data)[:batch_size]

        gen = list()
        dis = list()
        clf = list()
        trainX= list()
        for i,x in enumerate(Data):
            trainX.append(x[0])
            gen.append(x[1])
            dis.append(x[2])
            clf.append(x[3])

        return
        {
            'input' : trainX,
            'gen' : gen,
            'dis' : dis,
            'clf' : clf
        }


    def update_centroids(self, Spoints, Tpoints):
        """
            Spoints, Tpoints are 
        """
        g= self.gamma 
        loss=0
        for i in range(self.nb_class):
            Cs = 1 / len(Spoints[i]) * sum(Spoints[i])
            Ct = 1 / len(Tpoints[i]) * sum(Tpoints[i])
            self.CenterS[i] =  g * self.centerS[i] + (1-g) * Cs
            self.CenterT[i] =  g * self.centerT[i] + (1-g) * Ct
            loss += np.power(self.CenterS[i]-self.CenterT[i])**2
        self.sem_loss = loss


    def update_centers(self, Data):
        """
            

        """
        Tpoints = [list() for i in range(self.nb_class)]
        Spoints = [list() for i in range(self.nb_class)]

        pred = self.model.predict_on_batch(Data["input"])
        for i, p in enumerate(pred):
            if Data[i][2] == [-1]:
                Tpoints[np.argmax(p[2])].append(Data['input'][i])
            else :
                Spoints[Data["clf"]][i].append(Data["input"][i])

        self.update_centroids(Spoints, Tpoints)



    def train_dataset(SourceXY, TargetXY, test, epochs, batch_size):
        """
            TargetY may be void, only for metrics usage
        """


        self.batch_size = batch_size

        for i in range(epochs):
            data = gen_batch(SourceXY, TargetXY,batch_size)
            self.update_centroids(data)
            self.model.train_on_batch(data["input"], data)



