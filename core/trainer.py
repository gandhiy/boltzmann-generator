import os
import tensorflow as tf
import tensorflow_probability as tfp



tfd = tfp.distributions
tfb = tfp.bijectors



class Trainer:
    def __init__(self, model, targets, verbose=0):
        self.model = model
        self.verbose = verbose
        if(self.verbose==1):
            print(self.model.summary())
        
        
        _targets = tf.random.shuffle(targets)
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices(_targets)
            .shuffle(len(targets)).batch(256)
        )

        self.epoch = 0
        self.iteration = 0



    def train(self, epochs):
        for epoch in range(epochs):            
            for i, target in enumerate(self.train_dataset):
                self.iteration += 1 
                self.model.train(target, (epoch, self.iteration, i))



