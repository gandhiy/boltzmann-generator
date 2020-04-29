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
        pass

    def ml_loss(self, c1=1.0):
        loss = MLLoss(c1)
        return loss.lossFunction

    def kl_loss(self, simulation, c1 = 1.0, ndims = 2, ehigh = 1e5, emax = 1e10):
        loss = KLLoss(c1, simulation, ndims, ehigh, emax)
        return loss.lossFunction

    def ml_kl_loss(self, simulation, c1 = 1.0, ndims = 2, ehigh=1e5, emax = 1e10, turnover=200):
        loss = MLKL(c1, simulation, ndims, ehigh, emax, turnover)
        return loss.lossFunction
        
    def rc_kl_loss(self, simulation, rc_func, vmin, vmax, c1=1.0,  ndims = 2, ehigh=1e5, emax=1e10, turnover=200):
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
        return self.c1 * tf.reduce_mean(energies + model.bijector.inverse_log_det_jacobian(real_space, self.n))


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
        
        

