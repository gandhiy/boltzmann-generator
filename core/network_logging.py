import tensorflow as tf

from network_base import Network
from pdb import set_trace as debug


class Logging(Network):
    def __init__(self, decorated_model):
        self.decorated_model = decorated_model
        self.log = self.decorated_model.log
        self.batch_iteration = self.decorated_model.batch_iteration
        self.training_iteration = self.decorated_model.training_iteration
        self.epoch = self.decorated_model.epoch
        self.log = self.decorated_model.log
        self.loss_value = self.decorated_model.loss_value

    def summary(self):
        self.decorated_model.summary()

    def forward_sample(self, n):
        return self.decorated_model.forward_sample(n)

    def backward_sample(self, targets):
        return self.decorated_model.backward_sample(targets)

    def train(self, x, step):
        self.decorated_model.train(x, step)
        self.update()

    def get_state(self):
        return self.decorated_model.get_state()

    def update(self):
        self.log.update(self.get_state())
        self.decorated_model.state = {}
    


class LogLoss(Logging):
    def __init__(self, decorated_model):
        Logging.__init__(self, decorated_model)
        
    def get_state(self):
        s = self.decorated_model.get_state()
        s['training/loss'] = (self.decorated_model.loss_value, self.decorated_model.training_iteration)
        return s

class LogTargetPlot(Logging):
    def __init__(self, decorated_model):
        Logging.__init__(self, decorated_model)

    def get_state(self):
        s = self.decorated_model.get_state()
        if(self.decorated_model.batch_iteration == 0):
            target = self.forward_sample(1000)
            s['training/forward_sample'] = (target, self.decorated_model.epoch)
        return s

class LogGaussPlot(Logging):
    def __init__(self, decorated_model):
        Logging.__init__(self, decorated_model)

    def get_state(self):
        s = self.decorated_model.get_state()
        if(self.decorated_model.batch_iteration == 0):
            target = self.forward_sample(1000)
            gauss = self.backward_sample(target)
            s['training/backward_sample'] = (gauss, self.decorated_model.epoch)
        return s
