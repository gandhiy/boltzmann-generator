import tensorflow as tf 



class getOpt:
    def __init__(self):
        pass
    
    def adam(self, lr=0.01, b1=0.99, b2=0.9, **kwargs):
        opt = AdamOpt(lr, b1, b2, **kwargs)
        return opt.optimizer()

    def rmsprop(self, lr=0.01, rho=0.9, momentum=0.0, **kwargs):
        opt = RMSPropOpt(lr, rho, momentum, **kwargs)
        return opt.optimizer()

    def sgd(self, lr=0.01, nesterov=False, **kwargs):
        opt = SGDOpt(lr, nesterov, **kwargs)
        return opt.optimizer()

class optimizerInterface:
    def __init__(self):
        pass

    def optimizer(self):
        raise NotImplementedError

class AdamOpt(optimizerInterface):
    def __init__(self, learning_rate, beta_1, beta_2, **kwargs):
        super(AdamOpt, self).__init__()
        self.lr = learning_rate
        self.b1 = beta_1
        self.b2 = beta_2
        self.kwargs = kwargs

    def optimizer(self):
        return tf.optimizers.Adam(
            learning_rate=self.lr, 
            beta_1=self.b1,
            beta_2=self.b2,
            **self.kwargs
            )

class RMSPropOpt(optimizerInterface):
    def __init__(self, learning_rate, rho, momentum, **kwargs):
        super(RMSPropOpt, self).__init__()
        self.lr = learning_rate
        self.rho = rho
        self.mt = momentum
        self.kwargs = kwargs

    def optimizer(self):
        return tf.optimizers.RMSprop(
            learning_rate=self.lr,
            rho=self.rho,
            momentum=self.mt,
            **self.kwargs 
        )
    
class SGDOpt(optimizerInterface):
    def __init__(self, learning_rate, nesterov, **kwargs):
        super(SGDOpt, self).__init__()
        self.lr = learning_rate
        self.nesterov = nesterov
        self.kwargs = kwargs

    def optimizer(self):
        return tf.optimizers.SGD(
            learning_rate=self.lr,
            nesterov=self.nesterov,
            **self.kwargs
        )