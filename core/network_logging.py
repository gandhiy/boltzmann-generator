import sys
import visuals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from network_base import Network


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
        """
         Plot the scaler loss function at every single training iteration         
        """
        Logging.__init__(self, decorated_model)
        
    def get_state(self):
        s = self.decorated_model.get_state()
        s['training/loss'] = (self.decorated_model.loss_value, self.training_iteration)
        return s

class LogTargetPlot(Logging):
    def __init__(self, decorated_model, simulation=None, xlim = [-1.5, 1.5], ylim = [-0.5, 2.01], c='white'):
        """
         Plot the forward samples from the currently trained model. Uses the
         simulation to plot the central potential density function if available.

         PARAMETERS:
         * simulation: simulation used to generate the samples to train the model
         * xlim (array): min and max values for x-axis in plotting (ex. [min, max])
         * ylim (array): min and max values for y-axis in plotting (ex. [min, max])
         * c (string): color of the points for plotting the samples
        """

        Logging.__init__(self, decorated_model)
        self.sim = simulation
        self.xlim = xlim
        self.ylim = ylim
        self.color = c

    def get_state(self):
        s = self.decorated_model.get_state()
        if(self.batch_iteration == 0):
            target = self.forward_sample(2500).numpy().T
            fig = plt.figure(figsize=(12, 8), dpi = 100)
            if(self.sim):
                visuals.plot_2D_potential(self.sim.simulation.central_potential, xlim=self.xlim, ylim=self.ylim, cmap='jet')
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)
            plt.scatter(target[0], target[1], s=1.15, c=self.color)
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

class DimerAverageLocationPlot(Logging):
    def __init__(self, decorated_model):
        Logging.__init__(self, decorated_model)

    def get_state(self):
        s = self.decorated_model.get_state()
        if(self.batch_iteration == 0):
            targets = self.forward_sample(2500).numpy().reshape((-1, 36, 2)) # (2500 x 36 x 2)
            average_positions = np.mean(targets, axis=0)
            fig = plt.figure(figsize=(12, 8), dpi=100)
            plt.xlabel("Average x position")
            plt.ylabel("Average y position")
            plt.plot(average_positions[0:2, 0], average_positions[0:2, 1], c='red', linewidth=8/3)
            plt.plot(average_positions[:, 0], average_positions[:, 1], 'o', c='red', markersize=8)
            plt.close()
            s['training/average_dimer_position'] = (fig, self.epoch)
        return s

class FreeEnergyPlot(Logging):
    def __init__(self, decorated_model, simulation, RC_func, bins = 200, reshape=None):
        Logging.__init__(self, decorated_model)
        self.RC_func = RC_func
        self.simulation = simulation
        self.bins = bins
        self.reshape = reshape
    
    def get_state(self):
        s = self.decorated_model.get_state()
        if(self.batch_iteration == 0):
            rc_samples = []
            weights = []
            samples = self.forward_sample(2500).numpy()
            log_probs = self.flow.log_prob(samples, event_ndims=samples.shape[1]).numpy()
            if(len(log_probs.shape) > 1):
                log_probs = log_probs.diagonal()
            

            for (t, lp) in list(zip(samples,  log_probs)):
                rc_samples.append(self.RC_func(t))
                if(self.reshape):
                    t = t.reshape(self.reshape)
                weights.append(np.exp(-self.simulation.getEnergy(t) + lp))

            ## generate the histogram values
            counts, bins = np.histogram(rc_samples, weights=weights, bins=self.bins)
            probs = (counts / np.sum(counts)) + 1e-9
            bin_centers = (bins[:-1] + bins[1:])/2.0
            fig = plt.figure(figsize = (12, 8), dpi = 100)
            FE = -np.log(probs)
            plt.plot(bin_centers[FE < -np.log(1e-9)], FE[FE < -np.log(1e-9)])
            plt.close()
            ## transform the histogram values
            s['training/free_energy'] = (fig, self.epoch)

        return s

class Checkpointing(Logging):
    def __init__(self, decorated_model, freq = 1):
        Logging.__init__(self, decorated_model)
        self.freq = freq

    def get_state(self):
        if(self.epoch % self.freq == 0 and self.batch_iteration == 0):
            self.save()
        return self.decorated_model.get_state()

    