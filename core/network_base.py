# basic network architecture:
# https://github.com/MokkeMeguru/glow-realnvp-tutorial/blob/master/tips/RealNVP_tutorial_en.ipynb


import os 
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from abc import ABC
from losses import getLoss
from tb_writer import tensorboard_writer
from optimizers import getOpt
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU

tfd = tfp.distributions
tfb = tfp.bijectors

class NN(Layer): 
    """
    Base neural network block for building a single RealNVP layer
    """
    def __init__(self, in_shape, hidden_layers=[512,512], activation="relu", name="nn"):
        super(NN, self).__init__(name="nn")
        layer_list = []
        for i, hidden_layer_size in enumerate(hidden_layers):
            layer_list.append(Dense(hidden_layer_size, activation=activation, name=f'dense_{i}_1'))
            layer_list.append(Dense(hidden_layer_size, activation=activation, name=f'dense_{i}_2'))
        self.layer_list = layer_list
        self.log_s_layer = Dense(in_shape, activation='tanh', name='log_s')
        self.t_layer = Dense(in_shape, name='t')

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        log_s = self.log_s_layer(y)
        t = self.t_layer(y)
        return log_s, t 

class RealNVPLayer(tfp.bijectors.Bijector):
    """
    A basic RealNVP layer built using two neural network blocks (one for the
    forward sample and one for backward sample)

    """
    def __init__(self, model, in_shape, hidden_layers=[512,512], forward_min_event_ndims=1, validate_args: bool = False, name='real_nvp'):
        super(RealNVPLayer, self).__init__(validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name)
        self.in_shape = in_shape
        
        nn_layer = model(self.in_shape[-1] // 2, hidden_layers)
        nn_input_shape = self.in_shape.copy()
        nn_input_shape[-1] = self.in_shape[-1] // 2

        x = Input(nn_input_shape)
        log_s, t = nn_layer(x)
        self.nn = Model(x, [log_s, t], name="nn")
    
    def _forward(self, x):
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_b = x_b
        log_s, t = self.nn(x_b)
        s = tf.exp(log_s)
        y_a = s * x_a + t
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        y_a, y_b = tf.split(y, 2, axis=-1)
        x_b = y_b
        log_s, t = self.nn(y_b)
        s = tf.exp(log_s)
        x_a = (y_a - t) / s
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        _, x_b = tf.split(x, 2, axis=-1)
        log_s, t = self.nn(x_b)
        return log_s

class Network(ABC):

    def summary(self):
        raise NotImplementedError

    def forward_sample(self, n):
        raise NotImplementedError

    def backward_sample(self, target):
        raise NotImplementedError
    
    def train(self, x):
        raise NotImplementedError

    
    def get_state(self):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError

    def save(self, name):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    

class RealNVP(Network, tf.Module):
    def __init__(self, loss, optimizer, chain_length = 6, in_shape = [2], 
        loc = [0.0, 0.0], scale = [1.0,1.0], nn_layers=[256, 256], model_name="temp"):

        """
         RealNVP network built with the specificed loss function and optimizer.

         PARAMETERS:
         * loss (losses.lossfunction) the loss function that points the gradients in
         the correct direction

         * optimizer (tensorflow.keras.optimizer) the gradient optimizer 

         * chain length (int) number of realnvp layers    
         * nn_layers (array) shape of neural network block
         * model_name (string) name to save model under 

         """ 
        super(RealNVP, self).__init__()
        self.model_name = model_name
        self.loss = loss
        self.opt = optimizer
        self.chain = []
        self.in_shape = in_shape
        self.nn_layers = nn_layers
        self.chain_length = chain_length

        if self.in_shape[-1] == 2:
            for _ in range(self.chain_length):
                self.chain.append(RealNVPLayer(NN, self.in_shape, self.nn_layers))
                self.chain.append(tfp.bijectors.Permute([1, 0]))
        
        else:
            for _ in range(self.chain_length):
                self.chain.append(
                    tfb.RealNVP(
                        num_masked=10,
                        shift_and_log_scale_fn=tfb.real_nvp_default_template(
                            hidden_layers=self.nn_layers,
                            name='realnvp_block'
                        )
                    )
                    
                )



        self.flow = tfd.TransformedDistribution(
            distribution=self.__generate_multivariate_normal(loc=loc, scale=scale),
            bijector=tfb.Chain(list(reversed(self.chain)))            
        )

        if model_name is not None:
            self.__set_up_logging(self.model_name)
        self.loss_value = 0
        self.state = {}
        self.epoch = 0
        self.batch_iteration = 0
        self.training_iteration = 0
        self.ckpt = tf.train.Checkpoint(model=self.flow)

    def __generate_multivariate_normal(self, loc=[0.], scale=[1.]):
        return tfd.MultivariateNormalDiag(loc, scale)

    def __set_up_logging(self, mn):
        self.save_path = f"../checkpoints/{mn}"
        os.makedirs(self.save_path, exist_ok=True)
        files = [f for f in os.listdir(self.save_path)]
        count = 0
        f = f'run_{count}'
        while f in files:
            count += 1
            f = f'run_{count}'
        self.save_path = os.path.join(self.save_path, f"{f}")

        os.makedirs(self.save_path, exist_ok=True)
        logs = os.path.join(self.save_path, 'tensorboard_logs')
        self.log = tensorboard_writer(tf.summary.create_file_writer(logs))

    def summary(self):
        """"
         prints a summary of the model architecture
        """
        for i,layer in enumerate(self.chain):
            x = Input([2])
            y = layer.forward(x)
            Model(x,y,name=f'layer_{i}_summary').summary()

    def forward_sample(self, n):
        """
         sample from the network-based distribution 

         PARAMETERS:
         * n (int): number of samples

         RETURNS:
         * samples (tensorflow.Tensor): samples from the distribution
        """
        return self.flow.sample(n)

    def backward_sample(self, target):
        """
         backward sampling through the neural network transforms 
         forward samples back into the gaussian normal space

         PARAMETERS:
         * target (tensorflow.Tensor or numpy.ndarray): forward samples

         RETURNS:
         * backward_samples (tensorflow.Tensor): samples for the input distribution
        """
        return self.flow.bijector.inverse(target)

    def train(self, x, step):       
        """
         A single training step on the realNVP network

         PARAMETERS:
         * x (tensorflow.Tensor): a minibatch of samples
         * step (tuple - (int, int, int)): epoch, training iteration, and batch
         iteration 

        """ 
        self.epoch, self.training_iteration, self.batch_iteration = step
        
        # calculate loss
        with tf.GradientTape() as tape:
            self.loss_value = self.loss(self.flow, x)
        
        grads = tape.gradient(self.loss_value, self.flow.trainable_variables)
        grads = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(self.flow.trainable_variables, grads)]

        self.opt.apply_gradients(zip(grads, self.flow.trainable_variables))
        self.state.update(self.get_state())

    def save(self, name=None):
        """
         save the current state of the model

         * name (string): folder name
        """
        if name is None:
            name = f"saved_models/epoch_{self.epoch + 1}/ckpt"
        save_path = os.path.join(self.save_path, name)
        return self.ckpt.save(save_path)

    def load(self, path):
        """
         Load in checkpoint for a model. Ignore 
         extensions, just use [path]/[name]

         * path
        """

        self.ckpt.restore(path).assert_consumed()

    def get_state(self):
        return {}

    def update(self):
        self.log.update(self.get_state())
        self.state = {}

