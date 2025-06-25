import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense, Reshape, GlobalAveragePooling1D, Dropout
from tensorflow.keras.layers import (
    Input, Embedding, GlobalAveragePooling1D, Concatenate, Dense,
    Multiply, Lambda, Layer, Dot, BatchNormalization
)
import tensorflow.keras.backend as K


def l2_normalize(x, axis=-1, epsilon=1e-10):
    return x / (tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keepdims=True)) + epsilon)


class DinAttention(Layer):
    def __init__(self, **kwargs):
        super(DinAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1', shape=(input_shape[0][-1], input_shape[0][-1]),
                                  initializer='glorot_uniform', trainable=True)
        self.W2 = self.add_weight(name='W2', shape=(input_shape[1][-1], input_shape[0][-1]),
                                  initializer='glorot_uniform', trainable=True)
        self.V = self.add_weight(name='V', shape=(input_shape[0][-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        super(DinAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs[0]: 目标特征 (batch_size, emb_dim)
        # inputs[1]: 历史序列特征 (batch_size, max_len, emb_dim)
        target = inputs[0]
        history = inputs[1]
        # 计算注意力得分
        score = tf.tanh(tf.matmul(target, self.W1) + tf.matmul(history, self.W2))
        score = tf.matmul(score, self.V)
        attention_weights = tf.nn.softmax(score, axis=1)
        output = tf.reduce_sum(history * attention_weights, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])