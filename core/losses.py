import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp

tfd = tfp.distributions


class getLoss:
    def __init__(self):
        pass

    def maximum_likelihood_loss(self, c1=1.0):
        loss = MLLoss(c1)
        return loss.lossFunction

    def kl_loss(self, simulation, c1 = 1.0, ndims = 2):
        loss = KLLoss(c1, simulation, ndims)
        return loss.lossFunction
        
class lossInterface:
    def __init__(self):
        pass

    def lossFunction(self, model, samples):
        raise NotImplementedError



class MLLoss(lossInterface):
    def __init__(self, c1):
        super(MLLoss, self).__init__()
        self.c1 = c1

    def lossFunction(self, model, samples):
        return -self.c1 * tf.reduce_mean(model.log_prob(samples))


class KLLoss(lossInterface):
    def __init__(self, c1, simulation, ndims):
        super(KLLoss, self).__init__()
        self.c1 = c1
        self.u = simulation.getEnergy
        self.n = 2
        

    def lossFunction(self, model, samples):
        real_space = model.sample(1000)
        gauss_samples = model.distribution.sample(1000)
        energies = tf.convert_to_tensor(np.array([self.u(np.expand_dims(s, axis=0)) for s in real_space],dtype=np.float32))
        return self.c1 * tf.reduce_mean(energies + model.bijector.forward_log_det_jacobian(gauss_samples, self.n))




class permutation_invariance_loss(lossInterface):
    def __init__(self, c1, number_of_permutes):
        super(permutation_invariance_loss, self).__init__()
        self.c1 = c1
        self.number_of_permutes = number_of_permutes

    def lossFunction(self, model, samples):
        total_loss = 0
        # generate a set of permutation off of the samples
        # calculate an average loss based on the permutations using
        # 

        return super().lossFunction(model, samples)
