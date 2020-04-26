import sys
import visuals
import tensorflow as tf
<<<<<<< HEAD
import numpy as np
=======
import matplotlib.pyplot as plt


>>>>>>> e531695f271da080f0f9571ecfb5d92ad8050134
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
    def __init__(self, decorated_model, sim=None, xlim = [-1.5, 1.5], ylim = [-0.5, 2.01]):
        Logging.__init__(self, decorated_model)
        self.sim = sim
        self.xlim = xlim
        self.ylim = ylim

    def get_state(self):
        s = self.decorated_model.get_state()
        if(self.batch_iteration == 0):
            target = self.forward_sample(2500).numpy().T
            fig = plt.figure(figsize=(12,8))
            if(self.sim):
                visuals.plot_2D_potential(self.sim.central_potential, xlim=self.xlim, ylim=self.ylim)
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)
            plt.scatter(target[0], target[1], s=0.85, c='black')
            plt.close()
            s['training/forward_samples'] = (fig, self.epoch)
        return s

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

<<<<<<< HEAD
class FreeEnergyPlot(Logging):
    def __init__(self, decorated_model, simulation, RC_func, bins = 200):
        Logging.__init__(self, decorated_model)
        self.RC_func = RC_func
        self.simulation = simulation
        self.bins = bins
    def get_state(self):
        s = self.decorated_model.get_state()
        if(self.batch_iteration == 0):
            rc_samples = []
            weights = []
            for t in self.forward_sample(2500).numpy():
                rc_samples.append(self.RC_func(t))
                w = np.exp(-self.simulation.calculate_energy(t) + \
                     self.flow.log_prob(t) + \
                     self.flow.bijector.forward_log_det_jacobian(t))
                weights.append(w)
            ## generate the histogram values
            counts, bins = np.histogram(rc_samples, weights=weights, bins=self.bins)
            probs = counts / np.sum(counts)
            bin_centers = (bins[:-1] + bins[1:])/2.0
            fig = plt.figure(figsize = [12, 8], dpi = 150)
            FE = -np.log(probs)
            plt.plot(bin_centers, FE)
            ## transform the histogram values
            s['training/free_energy'] = (fig, self.epoch)
=======

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
>>>>>>> e531695f271da080f0f9571ecfb5d92ad8050134




class Checkpointing(Logging):
    def __init__(self, decorated_model, freq = 1):
        Logging.__init__(self, decorated_model)
        self.freq = freq

    def get_state(self):
        if(self.epoch % self.freq == 0 and self.batch_iteration == 0):
            self.save()
        return self.decorated_model.get_state()

    