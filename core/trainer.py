import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


from data import gen_double_moon_samples
from networks import RealNVP

from pdb import set_trace as debug


tfd = tfp.distributions
tfb = tfp.bijectors


class Trainer:
    def __init__(self, configs, verbose=0):
        self.model = RealNVP(**configs)
        
        if(verbose==1):
            print(self.model.summary())
        
        _targets = gen_double_moon_samples(10000)
        _targets = tf.random.shuffle(_targets)
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices(_targets)
            .shuffle(5000).batch(5000)
        )


    def _test_function(self):
        pass

    def train(self, epochs):
        for epoch in range(epochs):
            for target in self.train_dataset:
                self.model.train(target)





if __name__ == "__main__":
    t = Trainer({
        'chain_length': 4,
        'in_shape': [2],
        'loss': 'basic',
        'loss_parameters': {
            'c1': 1.0
        },
        'optimizer': 'Adam',
        'loc': [0.5, -2.5],        
    })
    t.train(10)