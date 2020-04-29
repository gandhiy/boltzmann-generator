import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp

from scipy import stats

tfd = tfp.distributions

class lossInterface:
    def __init__(self):
        pass

    def lossFunction(self, model, samples):
        raise NotImplementedError

class getLoss:
    def __init__(self):
        """
         Get the desired loss function
        """
        pass

    def ml_loss(self, c1=1.0):
        """
         returns the maximum likelihood loss based on the negative log
         likelihood
         
         PARAMETERS:
         c1: a weighting factor on the loss
        """
        loss = MLLoss(c1)
        return loss.lossFunction

    def kl_loss(self, simulation, c1 = 1.0, ndims = 2, ehigh = 1e5, emax = 1e10):
        """
         returns the kl loss that measures the energy of the forward samples and
         the log determinant jacobian of samples from the latent space
         distribution
         
         PARAMETERS:
         * simulation: a simulation object 
         * c1: loss value weight
         * ndims: number of dimensions for sampling
         * ehigh: value to scale energy values against
         * emax: maximum energy value to avoid exponential blowups
        """
        loss = KLLoss(c1, simulation, ndims, ehigh, emax)
        return loss.lossFunction

    def ml_kl_loss(self, simulation, c1 = 1.0, ndims = 2, ehigh=1e5, emax = 1e10, turnover=200):
        """
         Combines the ml and kl loss by first using the ml loss for a given
         number of training iterations and then adding in the kl loss

         PARAMETERS:
         * simulation: a simulation object
         * c1: loss value weight
         * ndims: dimensionality of the problem
         * ehigh: value to scale energy values against
         * emax: maximum energy value to avoid exponential blowups
         * turnover: number of training iterations before using kl_loss
        """
        loss = MLKL(c1, simulation, ndims, ehigh, emax, turnover)
        return loss.lossFunction
        
    def rc_kl_loss(self, simulation, rc_func, vmin, vmax, c1=1.0,  ndims = 2, ehigh=1e5, emax=1e10, turnover=200):
        """
         Combines the kl loss and a reaction coordinate loss that measures a
         batch-wise kernel density on the reaction coordinates. For the first
         number of training iterations, though, the network uses the maximum
         likelihood loss.

         PARAMETERS:
         * simulation: a simulation object
         * rc_func: a function that maps configuration samples to reaction
           coordinates
         * vmin: the minimum reaction coordinate value 
         * vmax: the maximum reaction coordinate value 
         * c1: loss weight factor
         * ndims: dimensionality of the sampling distribution
         * ehigh: value to scale energy values against
         * emax: maximum energy value to avoid exponential blowups
         * turnover: number of training iterations before adding kl and rc losses
        """

        loss = RCKL(c1, simulation, rc_func, ndims, vmin, vmax, ehigh, emax, turnover)
        return loss.lossFunction

class MLLoss(lossInterface):
    def __init__(self, c1):
        super(MLLoss, self).__init__()
        self.c1 = c1

    def lossFunction(self, model, samples):
        return -self.c1 * tf.reduce_mean(model.log_prob(samples))

class KLLoss(lossInterface):
    def __init__(self, c1, simulation, ndims, ehigh = 1e5, emax = 1e10):
        super(KLLoss, self).__init__()
        self.c1 = c1
        self.u = simulation.getEnergy
        self.n = ndims
        self.e_high = ehigh
        self.e_max = emax
    
    def scale_energy(self, e):
        if e < self.e_high:
            return e
        elif (e >= self.e_high) and (e < self.e_max):
            return self.e_high + np.log( min(e, self.e_max) - self.e_high + 1)

    def lossFunction(self, model, samples):
        gauss_samples = model.distribution.sample(len(samples))
        real_space = model.bijector.forward(gauss_samples)
        energies = tf.convert_to_tensor(np.array([self.scale_energy(self.u(np.expand_dims(s, axis=0))) for s in real_space],dtype=np.float32))
        return self.c1 * tf.reduce_mean(energies - model.bijector.forward_log_det_jacobian(gauss_samples, self.n))

class RCKL(KLLoss):
    def __init__(self, c1, simulation, RC_func, ndims, vmin, vmax, ehigh=1e5, emax=1e10, turnover = 200):
        super(RCKL, self).__init__(c1, simulation, ndims, ehigh, emax)
        self.r = RC_func
        self.min = vmin
        self.max = vmax
        self.count = 0
        self.turnover=turnover
    
    def reaction_coords(self, x):
        rx =  self.r(x)
        if(rx < self.min):
            return self.min
        elif(rx > self.max):
            return self.max
        return rx

    def lossFunction(self, model, samples):
        if(self.count < self.turnover):
            return -self.c1 * tf.reduce_mean(model.log_prob(samples))
        else:
            real_space = model.sample(len(samples))
            rx = np.array([self.reaction_coords(x) for x in real_space], dtype=np.float32)
            kde = tf.convert_to_tensor(stats.gaussian_kde(rx).logpdf(rx))
            return super().lossFunction(model, samples) + tf.reduce_mean(kde) - tf.reduce_mean(model.log_prob(samples))

class MLKL(KLLoss):
    def __init__(self, c1, simulation, ndims, ehigh = 1e5, emax = 1e10, turnover=200):
        super(MLKL, self).__init__(c1, simulation, ndims, ehigh, emax)
        self.count = 0
        self.turnover = turnover

    def lossFunction(self, model, samples):
        if(self.count < self.turnover):
            return -self.c1 * tf.reduce_mean(model.log_prob(samples))
        else:    
            return super().lossFunction(model, samples) - self.c1 * tf.reduce_mean(model.log_prob(samples))
        self.count += 1
        
        

