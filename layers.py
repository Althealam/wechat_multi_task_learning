import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense, Reshape, GlobalAveragePooling1D, Dropout

def _apply_attention(inputs):
    """使用标准层实现注意力机制，避免Lambda层潜在问题"""
    # 计算注意力分数 (batch_size, seq_len, 1)
    attn_scores = tf.keras.layers.Dense(1, activation='tanh')(inputs)
    # 在序列长度维度上应用softmax (axis=1)
    attn_weights = tf.keras.layers.Softmax(axis=1)(attn_scores)
    # 加权求和 (batch_size, hidden_dim)
    context = tf.keras.layers.Multiply()([inputs, attn_weights])
    context = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    return context

# 位置编码实现
def positional_encoding(length, depth):
    depth = depth/2
    positions = tf.range(length)[:, tf.newaxis]
    depths = tf.range(depth)[tf.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], -1)
    return tf.cast(pos_encoding, tf.float32)