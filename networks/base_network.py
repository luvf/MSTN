from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D,Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D



class base_network(object):
    """docstring for abstract_network"""
    def __init__(self):
        super(base_network, self).__init__()
    
    

    def gen_generator(self,input):
        #enc = Model()
        
        x = Conv2D(128,  kernel_size=(5, 5), strides=(2, 2), activation='relu')(input)
        x = MaxPooling2D( pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(64,  kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = MaxPooling2D( pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(32,  kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
        x = MaxPooling2D( pool_size=(2, 2), strides=(2, 2))(x)
        #modelx= Flatten()
        x = Dense(4096, activation='relu', name='gen')(x)
        
        #model = Model(input, x, name='generator')
        #model.summary()
        return x
        #noise = Input(shape=(self.latent_dim,))
        #img = model(noise)

        

    def gen_discriminor(self, input):       
        #disc = Sequential() 
        x = Dense(4096, activation='relu', name='fc3')(input)
        x = Dense(4096, activation='relu', name='fc4')(x)
        #x = Conv1D(64,  kernel_size=5,name = "cc1", strides=1, activation='relu')(input)
        #x = MaxPooling1D( pool_size=2 ,name = "cc2", strides=1)(x)
        #x = Conv2D(32,  kernel_size=(5, 5),name = "cc3", strides=(2, 2), activation='relu')(x)
        #x = MaxPooling2D( pool_size=(2, 2),name = "cc4", strides=(2, 2))(x)
        #x = Conv2D(64,  kernel_size=(5, 5),name = "cc5", strides=(2, 2), activation='relu')(x)
        #x = MaxPooling2D( pool_size=(2, 2),name = "cc8", strides=(2, 2))(x)
        #x = Dense(512, activation='relu', name='fc1')(x)
        output = Dense(1, name='dis')(x)

        #disc.summary()
        #model = Model(input, output, name='discriminator', )
        #model.summary()
        return output


        

    def gen_classifier(self, input, output_dim):
        #clf = Sequential()

        x = Dense(4096, activation='relu', name='fc1')(input)
        x = Dense(4096, activation='relu', name='fc2')(x)
        output = Dense(output_dim, activation='softmax', name='clf')(x)

        #model = Model(input, x, name='classifier')
        #model.summary()
        return output
        #clf.summary()
