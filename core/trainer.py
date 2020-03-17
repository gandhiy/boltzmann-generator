import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


from networks import RealNVP

tfd = tfp.distributions
tfb = tfp.bijectors


class Trainer:
    def __init__(self, configs, verbose=0):
        self.model = RealNVP(**configs)
        self.flow = self.model.flow
        if(verbose==1):
            print(self.model.summary())
        

    def sample(self):
        samples = self.flow.sample(10000)
        plt.figure(figsize=(8,6))
        plt.xlim([-4, 4])        
        plt.ylim([-4, 4])
        plt.scatter(samples[:, 0], samples[:, 1], s=15)
        plt.show()



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
    t.sample()