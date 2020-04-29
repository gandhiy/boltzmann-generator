import os
import tensorflow as tf
import tensorflow_probability as tfp



tfd = tfp.distributions
tfb = tfp.bijectors



class Trainer:
    def __init__(self, model, targets, verbose=0, batch_size = 1000):
        """
         Trainer class wrapper. Wraps the model and data into one easy to use
         class that can be called.

         PARAMETERS:
         * model: Network to train (decorated or not)
         * targets: data set of samples to train on (# of samples x # of
           features)
         * verbose: if equal to 1 will print a summary of the model
         * batch_size: number of samples in a mini-batch
        """
        self.model = model
        self.verbose = verbose
        if(self.verbose==1):
            print(self.model.summary())
        
        
        _targets = tf.random.shuffle(targets)
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices(_targets)
            .shuffle(len(targets)).batch(batch_size)
        )

        self.epoch = 0
        self.iteration = 0


    def train(self, epochs):
        """
         Training function for the given model on the given data.

         PARAMETERS:
         * epochs: number of epcohs to run on all batches 
        """
        for epoch in range(epochs):            
            for i, target in enumerate(self.train_dataset):
                self.iteration += 1 
                self.model.train(target, (epoch, self.iteration, i))



