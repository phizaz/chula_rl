from collections import deque
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers, optimizers

from chula_rl.policy.base_policy import BasePolicy

# import torch
# import torch.nn.functional as F
# from torch import distributions, nn, optim


@dataclass
class A2CConf:
    discount_factor: float
    lr: float
    c_v: float
    c_ent: float
    clip_grad: float


class A2CPolicy(BasePolicy):
    def __init__(self, s_dim, a_dim, n_hid, device, conf: A2CConf):
        self.device = device
        self.conf = conf

        self.v = ValueNet(s_dim, n_hid)
        self.pi = PolicyNet(s_dim, a_dim, n_hid)

        self.opt_v = optimizers.SGD(conf.lr)
        self.opt_pi = optimizers.SGD(conf.lr)

        # used for statistics
        self.v_mean = deque(maxlen=30)
        self.std = deque(maxlen=100)
        self.ent = deque(maxlen=30)

    def get_stats(self):
        return {
            'policy/v_mean': np.array(self.v_mean).mean(),
            'policy/std': np.array(self.std).mean(),
            'policy/ent': np.array(self.ent).mean(),
        }

    def step(self, state):
        s = tf.convert_to_tensor(state)
        mean, logstd = self.pi(s)
        std = tf.math.exp(logstd)
        d = tfp.distributions.Normal(mean, std)
        # samples actions
        a = d.sample()
        self.std.append(tf.reduce_mean(std).numpy())
        return a.numpy()

    def seq_v(self, s):
        """flatten s and then query for v, used during optimization"""
        t, b = s.shape[0], s.shape[1]
        v = self.v(tf.reshape(s, [t * b, -1]))
        return tf.reshape(v, [t, b])

    def seq_pi(self, s):
        """flatten s and then query for a, used during optimization"""
        t, b = s.shape[0], s.shape[1]
        mean, logstd = self.pi(tf.reshape(s, [t * b, -1]))
        mean = tf.reshape(mean, [t, b, -1])
        logstd = tf.reshape(logstd, [t, b, -1])
        # a distribution object could be used to calculate log_prob, entropy etc...
        d = tfp.distributions.Normal(mean, tf.math.exp(logstd))
        return d

    def optimize_step(self, data):
        def convert(x):
            return tf.convert_to_tensor(x)

        s = convert(data['s'])
        a = convert(data['a'])
        r = convert(data['r'])
        done = convert(data['done'])
        final_s = convert(data['final_s'])

        # target
        final_v = self.v(final_s)[..., 0]
        returns = discount_bootstrap_return(r, done, final_v,
                                            self.conf.discount_factor)

        with tf.GradientTape(persistent=True) as t:
            # value loss
            # (v - return)
            v = self.seq_v(s)
            loss_v = tf.reduce_mean(0.5 * (v - returns)**2)
            self.v_mean.append(tf.reduce_mean(v).numpy())

            # policy loss
            # (q-v) * log pi
            pi = self.seq_pi(s)
            adv = returns - tf.stop_gradient(v)
            loss_pg = -tf.reduce_mean(adv * pi.log_prob(a)[..., 0]) # remove the last dimension
            ent = tf.reduce_mean(pi.entropy())
            loss_ent = -ent
            self.ent.append(ent.numpy())

            # total loss function
            loss = (self.conf.c_v * loss_v + loss_pg +
                    self.conf.c_ent * loss_ent)

        # gradients
        grads_v = t.gradient(loss, self.v.trainable_weights)
        grads_pi = t.gradient(loss, self.pi.trainable_weights)

        # gradient clipping (very important)
        grads_v, _ = tf.clip_by_global_norm(grads_v, self.conf.clip_grad)
        grads_pi, _ = tf.clip_by_global_norm(grads_pi, self.conf.clip_grad)

        # optmizer step
        self.opt_v.apply_gradients(zip(grads_v, self.v.trainable_weights))
        self.opt_pi.apply_gradients(zip(grads_pi, self.pi.trainable_weights))


class ValueNet(keras.Model):
    """value network"""
    def __init__(self, s_dim, n_hid):
        super().__init__()
        self.s_dim = s_dim
        self.n_hid = n_hid
        self.net = keras.Sequential([
            layers.Dense(n_hid, activation='relu', input_shape=(s_dim, )),
            layers.Dense(n_hid, activation='relu'),
            layers.Dense(1),
        ])

    def call(self, s):
        return self.net(s)


class PolicyNet(keras.Model):
    """policy network"""
    def __init__(self, s_dim, a_dim, n_hid):
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n_hid = n_hid
        self.net = keras.Sequential([
            layers.Dense(n_hid, activation='relu', input_shape=(s_dim, )),
            layers.Dense(n_hid, activation='relu'),
        ])
        self.mean = layers.Dense(a_dim, input_shape=(n_hid, ))
        self.logstd = layers.Dense(a_dim, input_shape=(n_hid, ))

    def call(self, s):
        h = self.net(s)
        mean, logstd = self.mean(h), self.logstd(h)
        return mean, logstd


def discount_bootstrap_return(
        r,
        done,
        final_v,
        discount_factor: float,
):
    """ Calculate state values bootstrapping off the following state values """
    assert len(final_v.shape) == 1, "we want final_v with no time dim"
    n_step = len(r)
    # returns = tf.zeros_like(r)
    returns = [None] * r.shape[0]
    # discount/bootstrap off value fn
    current_value = final_v
    for i in reversed(range(n_step)):
        current_value = r[i] + discount_factor * current_value * (
            1.0 - tf.cast(done[i], tf.float32))
        returns[i] = current_value
    returns = tf.stack(returns)
    return returns
