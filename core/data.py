import tensorflow as tf
import tensorflow_probability as tfp 

tfd = tfp.distributions


def gen_double_moon_samples(num_samples):
    x1_1 = tfd.Normal(loc=4.0, scale=4.0)
    x1_1_samples = x1_1.sample(num_samples // 2)
    x1_2 = tfd.Normal(
        loc=0.25 * (x1_1_samples - 4) ** 2 - 20, 
        scale=tf.ones_like(num_samples / 2) * 2
    )
    x1_2_samples = x1_2.sample()

    x2_1 = tfd.Normal(loc=4.0, scale=4.0)
    x2_1_samples = x2_1.sample(num_samples // 2)
    x2_2 = tfd.Normal(
        loc=-0.25 * (x2_1_samples - 4) ** 2 + 20,
        scale=tf.ones_like(num_samples / 2) * 2,
    )
    x2_2_samples = x2_2.sample()
    
    x1_samples = tf.stack([x1_1_samples * 0.2, x1_2_samples * 0.1], axis=1)
    x2_samples = tf.stack([x2_1_samples * 0.2 - 2, x2_2_samples * 0.1], axis=1)

    x_samples = tf.concat([x1_samples, x2_samples], axis=0)
    return x_samples

