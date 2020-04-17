import tensorflow as tf 



class getLoss:
    def __init__(self):
        pass

    def basic_loss(self, c1=1.0):
        loss = basicLoss(c1)
        return loss.lossFunction


        
class lossInterface:
    def __init__(self):
        pass

    def lossFunction(self, target_prob, samples):
        raise NotImplementedError


class basicLoss(lossInterface):
    def __init__(self, c1):
        super(basicLoss, self).__init__()
        self.c1 = c1

    
    def lossFunction(self, target_prob, samples):
        return -self.c1 * tf.reduce_mean(target_prob)


