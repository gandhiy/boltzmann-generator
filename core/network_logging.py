import tensorflow as tf

from network_base import Network
from pdb import set_trace as debug


class Logging(Network):
    def __init__(self, decorated_model):
        self.decorated_model = decorated_model
        self.log = self.decorated_model.log
        self.log = self.decorated_model.log
        self.loss_value = self.decorated_model.loss_value

    def summary(self):
        self.decorated_model.summary()

    def forward_sample(self, n):
        return self.decorated_model.forward_sample(n)

    def backward_sample(self, targets):
        return self.decorated_model.backward_sample(targets)

    def train(self, x, step):
        self.epoch, self.training_iteration, self.batch_iteration = step
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
        s['training/loss'] = (self.decorated_model.loss_value, self.training_iteration)
        return s

class LogTargetPlot(Logging):
    def __init__(self, decorated_model):
        Logging.__init__(self, decorated_model)

    def get_state(self):
        s = self.decorated_model.get_state()
        if(self.batch_iteration == 0):
            target = self.forward_sample(1000)
            s['training/forward_sample'] = (target, self.epoch)
        return s

class LogGaussPlot(Logging):
    def __init__(self, decorated_model):
        Logging.__init__(self, decorated_model)

    def get_state(self):
        s = self.decorated_model.get_state()
        if(self.batch_iteration == 0):
            target = self.forward_sample(1000)
            gauss = self.backward_sample(target)
            s['training/backward_sample'] = (gauss, self.epoch)
        return s
