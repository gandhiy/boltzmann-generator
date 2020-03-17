import tensorflow as tf 

from tensorflow.keras.optimizers import SGD, Adam, RMSprop

class getOpt:
    def __init__(self, type):
        self.type = type
    def get_optimizer(self, parameters = None):
        if(self.type == 'SGD'):
            return SGD(parameters)
        elif(self.type == 'Adam'):
            return Adam(parameters)
        elif(self.type == 'RMSprop'):
            return RMSprop(parameters)
        else:
            raise ValueError("need to specificy a corrent optimizer type")
            
