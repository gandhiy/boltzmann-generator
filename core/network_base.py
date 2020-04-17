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

from pdb import set_trace as debug

class NN(Layer): 
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
    
    def train(self, x, step):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError
    
    def update(self):
        raise NotImplementedError

    
class RealNVP(Network):
    def __init__(self, loss, optimizer, chain_length = 6, in_shape = [2], 
        nn_layers=[256, 256], loc=[0., 0.], scale=[1., 1.],
        model_name='temp'):
        super(RealNVP, self).__init__()

        self.loc = loc
        self.scale = scale
        self.model_name = model_name
        self.loss = loss
        self.opt = optimizer
        self.chain = []
        self.in_shape = in_shape
        self.nn_layers = nn_layers
        self.chain_length = chain_length


        for _ in range(self.chain_length):
            self.chain.append(RealNVPLayer(NN, self.in_shape, self.nn_layers))
            self.chain.append(tfp.bijectors.Permute([1, 0]))
        
        self.flow = tfd.TransformedDistribution(
            distribution=self.__generate_multivariate_normal(self.loc, self.scale),
            bijector=tfb.Chain(list(reversed(self.chain)))
        )

        self.__set_up_logging(self.model_name) 
        self.state = {}
        self.epoch = 0
        self.batch_iteration = 0
        self.training_iteration = 0
        self.loss_value = 0

    def __generate_multivariate_normal(self, loc=[0., 0.], scale=[1., 1.]):
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
        for i,layer in enumerate(self.chain):
            x = Input([2])
            y = layer.forward(x)
            Model(x,y,name=f'layer_{i}_summary').summary()

    def forward_sample(self, n):
        return self.flow.sample(n)

    def backward_sample(self, target):
        return self.flow.bijector.inverse(target)

    def train(self, x, step):        
        self.epoch, self.training_iteration, self.batch_iteration = step
        
        # calculate loss
        with tf.GradientTape() as tape:
            self.loss_value = self.loss(self.flow.log_prob(x), self.flow.sample(x.shape[0]))
        
        grads = tape.gradient(self.loss_value, self.flow.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.flow.trainable_variables))
        

    def get_state(self):
        return {}

    def update(self):
        pass

