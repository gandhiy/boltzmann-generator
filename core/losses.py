import tensorflow as tf 



class getLoss:
    def __init__(self, type):
        self.type = type

    def get_loss(self, parameters):
        if(self.type == 'basic'):
            lossfunction = basicLoss(parameters)
            return lossfunction.lossfn()
        else:
            raise ValueError("need to specificy a corrent optimizer type")
            

class basicLoss:
    def __init__(self, parameters):
        self.parameters = parameters
    def lossfn(self):
        def loss(targets):
            return -self.parameters['c1'] * tf.reduce_mean(targets)
        return loss
