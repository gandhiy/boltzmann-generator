import tensorflow as tf
import tensorflow_probability as tfp

from losses import getLoss
from optimizers import getOpt
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, ReLU

tfd = tfp.distributions
tfb = tfp.bijectors


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

class RealNVP():
    def __init__(self, chain_length, in_shape, loss, loss_parameters, optimizer,
        optimizer_parameters=None, nn_layers=[256, 256], loc=[0., 0.], scale=[1., 1.]):
        
        
        self.loss = getLoss(loss).get_loss(loss_parameters)
        self.opt = getOpt(optimizer).get_optimizer(optimizer_parameters)
        self.chain = []
        self.in_shape = in_shape
        self.nn_layers = nn_layers
        for _ in range(chain_length):
            self.chain.append(RealNVPLayer(NN, self.in_shape, self.nn_layers))
            self.chain.append(tfp.bijectors.Permute([1, 0]))
        
        self.flow = tfd.TransformedDistribution(
            distribution=self.__generate_multivariate_normal(loc, scale),
            bijector=tfb.Chain(list(reversed(self.chain)))
        )

        self.avg_loss = tf.keras.metrics.Mean(name='average_loss', dtype=tf.float32)        
        self.log = tf.summary.create_file_writer('checkpoints')

    def __generate_multivariate_normal(self, loc=[0., 0.], scale=[1., 1.]):
        return tfd.MultivariateNormalDiag(loc, scale)


    def summary(self):
        for i,layer in enumerate(self.chain):
            x = Input([2])
            y = layer.forward(x)
            Model(x,y,name=f'layer_{i}_summary').summary()


    def forward_sample(self, n):
        return self.flow.sample(n)
    def backward_sample(self, target):
        return self.flow.bijector.inverse(target)



    def train(self, x):
        with tf.GradientTape as tape:
            log_prob_loss = self.loss(self.flow.log_prob(x))
        grads = tape.gradient(log_prob_loss, self.flow.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.flow.trainable_variables))
        self.avg_loss.update_state(log_prob_loss)
        if(tf.equal(self.opt.iterations % 100, 0)):
            with self.log.as_default():
                tf.summary.scalar("loss", self.avg_loss.result(), step=self.opt.iterations)
                self.avg_loss.reset_states()
        




def test():
    print("testing Feed Forward Network")
    nn = NN(1)
    x = Input([1])
    log_s, t = nn(x)
    Model(x, [log_s, t], name="nn_test").summary()

    
    
    print("testing RealNVPLayer")
    realnvp = RealNVPLayer(NN, in_shape=[2])
    x = Input([2])
    y = realnvp.forward(x)
    print("trainable_variables : ", len(realnvp.trainable_variables))
    Model(x, y, name='realnvp_test').summary()

    


if __name__ == "__main__":
    test()