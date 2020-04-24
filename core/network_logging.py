import tensorflow as tf

from network_base import Network
from pdb import set_trace as debug


class Logging(Network):
    def __init__(self, decorated_model):
        self.decorated_model = decorated_model
        self.log = self.decorated_model.log
        self.log = self.decorated_model.log
        self.loss_value = self.decorated_model.loss_value
        self.flow = self.decorated_model.flow

    def summary(self):
        self.decorated_model.summary()

    def forward_sample(self, n):
        return self.decorated_model.forward_sample(n)

    def backward_sample(self, targets):
        return self.decorated_model.backward_sample(targets)

    def save(self, name=None):
        return self.decorated_model.save(name)

    def load(self, path):
        self.decorated_model.load(path)
    
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
            target = self.forward_sample(2500)
            s['training/forward_sample'] = (target, self.epoch)
        return s

# class FreeEnergyPlot(Logging):
#     def __init__(self, decorated_model, RC_func):
#         Logging.__init__(self, decorated_model)
#         self.RC_func = RC_func

#     def get_state(self):
#         s = self.decorated_model.get_state()
#         if(self.batch_iteration == 0):
#             samples = []
#             for t in self.forward_sample(2500).numpy():
#                 samples.append(self.RC_func(t))
                
#             ## generate the histogram values
#             ## transform the histogram values

#             s['training/free_energy'] = (samples, self.epoch) 


class LogGaussPlot(Logging):
    def __init__(self, decorated_model):
        Logging.__init__(self, decorated_model)

    def get_state(self):
        s = self.decorated_model.get_state()
        if(self.batch_iteration == 0):
            target = self.forward_sample(2500)
            gauss = self.backward_sample(target)
            s['training/backward_sample'] = (gauss, self.epoch)
        return s


class Checkpointing(Logging):
    def __init__(self, decorated_model, freq = 1):
        Logging.__init__(self, decorated_model)
        self.freq = freq

    def get_state(self):
        if(self.epoch % self.freq == 0 and self.batch_iteration == 0):
            self.save()
        return self.decorated_model.get_state()

    