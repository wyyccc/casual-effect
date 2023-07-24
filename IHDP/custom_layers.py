import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import glorot_normal

try:
    from tensorflow.python.ops.init_ops_v2 import Zeros, Ones, glorot_normal
except ImportError:
    from tensorflow.python.ops.init_ops import Zeros, Ones, glorot_normal_initializer as glorot_normal

from tensorflow.python.keras.layers import Layer, Dropout

try:
    from tensorflow.python.keras.layers import BatchNormalization
except ImportError:
    BatchNormalization = tf.keras.layers.BatchNormalization
from tensorflow.python.keras.regularizers import l2

class EpsilonLayer(Layer):
    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]


class MMOELayer(Layer):
    def __init__(self, num_tasks, num_experts, output_dim, gate_dropout=0.0, seed=1024, **kwargs):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.gate_dropout = gate_dropout
        self.seed = seed
        super(MMOELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.expert_kernel = self.add_weight(
                                    name='expert_kernel',
                                    shape=(input_dim, self.num_experts * self.output_dim),
                                    dtype=tf.float32,
                                    initializer=glorot_normal(seed=self.seed))
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(self.add_weight(
                                        name='gate_weight_' + str(i),
                                        shape=(input_dim, self.num_experts),
                                        dtype=tf.float32,
                                        initializer=glorot_normal(seed=self.seed)))
        super(MMOELayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        outputs = []
        expert_out = tf.keras.layers.Dense(self.num_experts * self.output_dim, activation='relu')(inputs)
        expert_out = tf.reshape(expert_out, [-1, self.output_dim, self.num_experts])
        for i in range(self.num_tasks):
            gate_out = tf.keras.layers.Dense(self.num_experts, activation='softmax')(inputs)
            gate_out = tf.keras.layers.Dropout(self.gate_dropout, seed=self.seed)(gate_out, training=training)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=1), [1, self.output_dim, 1])
            output = tf.reduce_sum(tf.multiply(expert_out, gate_out), axis=2)
            outputs.append(output)
        return outputs

    def get_config(self):
        config = {'num_tasks': self.num_tasks,
                  'num_experts': self.num_experts,
                  'output_dim': self.output_dim}
        base_config = super(MMOELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_dim] * self.num_tasks


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='elu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads

    def call(self, q, k, v):
        """
        inputs: tensor(None, dk)  tensor(None, dk)  tensor(None, dv)
        output: tensor(None, dv)
        """
        self.dk, self.dv, self_att = q.shape[1], v.shape[1], []
        for _ in range(self.num_heads):
            query = tf.keras.layers.Dense(self.dv)(q)
            key = tf.keras.layers.Dense(self.dv)(k)
            value = tf.keras.layers.Dense(self.dv)(v)
            self_att.append(tf.multiply(tf.keras.layers.Softmax()(tf.multiply(query, key)) / np.sqrt(self.dv), value))
        self_att = tf.keras.layers.Concatenate(1)(self_att)
        self_att = tf.keras.layers.Dense(self.dv)(self_att) + v
        self_att = tf.keras.layers.Dense(self.dv)(tf.keras.layers.BatchNormalization()(self_att)) + self_att
        return self_att


if __name__ == '__main__':
    att_layer = AttentionLayer(4)
    tensor = tf.ones((100, 10))
    print(att_layer(tensor).shape)