from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras import losses

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)



class MSTN(object):
    """docstring for MSTN"""
    def __init__(self, input_dim, nb_class,   ):
        super(MSTN, self).__init__()
        self.nb_class = nb_class
        self.gamma = 0.3
        
        self.latent_space_dim = 200
        self.model = self.create_model()

        self.CenterS = np.zeros([nb_class,latent_space_dim])
        self.CenterT = np.zeros([nb_class,latent_space_dim])




    def create_model(input):
        
        losses = {
            "gen": self.sem_loss,
            "dis": self.D_loss,
            "clf": self.classification_losss,
            }


        losses_wheights = {
            'gen' : 1,
            'dis' : 1,
            'clf' : 1

        }



        model = Sequential()
        
        inputs = Input()
        gen = self.gen_generator(inputs, latent_space)

        dis = self.gen_discriminor(gen)
        
        clf = self.gen_classifier(gen)




        model = Model(inputs, clf, name='model', outputs = [gen, dis, clf])
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        return model
  
    def gen_generator(self,input):
        #enc = Sequential()
        
        x = Dense(4096, activation='relu', name='fc1')(input)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(4096, activation='relu', name='gen')(x)
        
        #enc.summary()

        #noise = Input(shape=(self.latent_dim,))
        #img = model(noise)

        
        return x

    def gen_discriminor(self, input):       
        #disc = Sequential() 

        x = Conv2D(16,  kernel_size=(5, 5), strides=(2, 2), activation='relu')(input)
        x = MaxPooling2D( pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(32,  kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = MaxPooling2D( pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(64,  kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = MaxPooling2D( pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = MaxPooling2D( pool_size=(2, 2), strides=(2, 2))(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dense(1, name='dis')(x)

        #disc.summary()



        return x
        

    def gen_classifier(self, input, output_dim):
        #clf = Sequential()

        x = Dense(4096, input, activation='relu', name='fc1')(input)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(output_dim, activation='softmax', name='clf')(x)
        
        #clf.summary()



        return x

 


    def D_loss(self,y_true,y_pred):
        return binary_crossentropy(y_true,y_pred)

    def C_loss(self, y_true, y_pred):
        pr = list()
        tr = list()
        for i in range(y_pred):
            if y_true != [-1]:
                pr.append(y_pred[i])
                tr.append(y_true[i])
        return K.losses.categorical_crossentropy(y_true,y_pred)

        
            

    def Sem_loss(self):
        return self.sem_loss / self.batch_size

    def gen_batch(Source, Target,batch_size):
        DataX = [(x, 0, 0, y)   for x, y in Source]
        DataY = [(x, 1, 0 ,[-1]) for x, y in Target]
        Data = X+Y
        np.random.shuffle(Data)[:batch_size]

        gen = list()
        dis = list()
        clf = list() 
        for i,x in enumerate(Data):
            trainX.append(x[0])
            #gen.append(x[1])
            dis.append(x[2])
            clf.append(x[3])

        return
        {
            'input' : TrainX,
            #'gen' : gen,
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
            self.CenterS[i] =  g * self.centerS[i] + (1-g) * CS
            self.CenterT[i] =  g * self.centerT[i] + (1-g) * CT
            loss+= np.power(CenterS[i]-CenterT[i])**2
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



