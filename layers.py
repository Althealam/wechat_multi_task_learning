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
        target = inputs[0]  # (batch_size, emb_dim)
        history = inputs[1] # (batch_size, max_len, emb_dim)
        max_len = tf.shape(history)[1]

        # 扩展target维度，和history在时间步上对齐
        target_expanded = tf.expand_dims(target, axis=1)  # (batch_size, 1, emb_dim)
        target_expanded = tf.tile(target_expanded, [1, max_len, 1])  # (batch_size, max_len, emb_dim)

        score = tf.tanh(tf.matmul(target_expanded, self.W1) + tf.matmul(history, self.W2))  # 都是 (batch_size, max_len, emb_dim)
        score = tf.matmul(score, self.V)  # (batch_size, max_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch_size, max_len, 1)

        output = tf.reduce_sum(history * attention_weights, axis=1)  # (batch_size, emb_dim)
        return output


    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])