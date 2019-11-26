from collections import deque
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers, optimizers

from chula_rl.policy.base_policy import BasePolicy


@dataclass
class DDPGConf:
    discount_factor: float
    lr: float
    explore_std: float
    clip_grad: float
    c_pi: float
    tau: float


class NetWithTarget(keras.Model):
    """support the use of slow changing target network"""
    def __init__(self, new, old):
        super().__init__()
        self.new = new
        self.old = old

    def call(self, *args, **kwargs):
        return self.new(*args, **kwargs)

    def update(self):
        # update old => new abruptly
        for p, p_old in zip(self.new.trainable_weights,
                            self.old.trainable_weights):
            p_old.assign(p)

    def update_polyak(self, tau: float):
        # slowly update the old => new
        for p, p_old in zip(self.new.trainable_weights,
                            self.old.trainable_weights):
            p_old.assign_add(tau * (p - p_old))


class DDPGPolicy(BasePolicy):
    """deep deterministic policy gradient algorithm"""
    def __init__(self, s_dim, a_dim, n_hid, device, conf: DDPGConf):
        self.device = device
        self.conf = conf

        self.q = NetWithTarget(QNet(s_dim, a_dim, n_hid),
                               QNet(s_dim, a_dim, n_hid))
        # set the old and new to be the same
        self.q.update()
        self.pi = NetWithTarget(DeterministicPolicyNet(s_dim, a_dim, n_hid),
                                DeterministicPolicyNet(s_dim, a_dim, n_hid))
        # set the old and new to be the same
        self.pi.update()

        # SGD is a good starting point
        self.opt_q = optimizers.SGD(conf.lr)
        self.opt_pi = optimizers.SGD(conf.lr)

        # only for statistics
        self.q_mean = deque(maxlen=30)
        self.a_mean = deque(maxlen=100)
        self.a_std = deque(maxlen=100)

    def get_stats(self):
        return {
            'policy/q_mean': np.array(self.q_mean).mean(),
            'policy/a_mean': np.array(self.a_mean).mean(),
            'policy/a_std': np.array(self.a_std).std(),
        }

    def step(self, state):
        """noise is added to explore"""
        s = tf.convert_to_tensor(state)
        a = self.pi.new(s)
        a = a + tf.random.normal(a.shape) * self.conf.explore_std
        self.a_mean.append(tf.reduce_mean(a).numpy())
        self.a_std.append(tf.math.reduce_std(a).numpy())
        return a.numpy()

    def optimize_step(self, data):
        if data is None: return

        # data could be of shape (batch, envs, *)
        # which needs to be flattened

        def convert(x):
            return tf.convert_to_tensor(x)

        # number of items
        m = data['s'].shape[0] * data['s'].shape[1]
        # usually reshape them into (m, -1)
        s = tf.reshape(convert(data['s']), (m, -1))
        ss = tf.reshape(convert(data['ss']), (m, -1))
        a = tf.reshape(convert(data['a']), (m, -1))
        r = tf.reshape(convert(data['r']), (m, ))
        done = tf.reshape(convert(data['done']), (m, ))

        # bootstrapped target value
        # note: we use "old" network
        aa = self.pi.old(ss)
        qq = self.q.old(tf.concat([ss, aa], axis=1))[..., 0] # remove the final dimension
        q_tgt = r + tf.cast(~done, tf.float32) * qq * self.conf.discount_factor
        assert len(q_tgt.shape) == 1

        with tf.GradientTape(persistent=True) as t:
            # value loss
            # (q - q_tgt)
            q = self.q.new(tf.concat([s, a], axis=1))[..., 0]
            assert len(q.shape) == 1
            loss_q = tf.reduce_mean(0.5 * (q - q_tgt)**2)
            self.q_mean.append(tf.reduce_mean(q).numpy())

            # policy loss
            # need to backpropagate through the policy
            a = self.pi.new(s)
            q = self.q.new(tf.concat([s, a], axis=1))[..., 0]
            loss_pg = tf.reduce_mean(-q)

            # total loss
            loss = loss_q + self.conf.c_pi * loss_pg

        # gradients calculated for each network
        grads_q = t.gradient(loss, self.q.new.trainable_weights)
        grads_pi = t.gradient(loss, self.pi.new.trainable_weights)

        # gradient clipping (very important)
        grads_q, _ = tf.clip_by_global_norm(grads_q, self.conf.clip_grad)
        grads_pi, _ = tf.clip_by_global_norm(grads_pi, self.conf.clip_grad)

        # optimize the network
        self.opt_q.apply_gradients(zip(grads_q, self.q.new.trainable_weights))
        self.opt_pi.apply_gradients(
            zip(grads_pi, self.pi.new.trainable_weights))

        # slowly update the target networks
        self.q.update_polyak(self.conf.tau)
        self.pi.update_polyak(self.conf.tau)


class QNet(keras.Model):
    """q network"""
    def __init__(self, s_dim, a_dim, n_hid):
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n_hid = n_hid
        self.net = keras.Sequential([
            layers.Dense(n_hid,
                         activation='relu',
                         input_shape=(s_dim + a_dim, )),
            layers.Dense(n_hid, activation='relu'),
            layers.Dense(1),
        ])

    def call(self, sa):
        return self.net(sa)


class DeterministicPolicyNet(keras.Model):
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
        # output the "mean" action
        # noise will be added later at exploration time
        self.mean = layers.Dense(a_dim, input_shape=(n_hid, ))

    def call(self, s):
        h = self.net(s)
        mean = tf.math.tanh(self.mean(h))
        return mean
