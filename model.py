# -*- coding: utf-8 -*-

from tensorflow.keras.layers import *
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import TruncatedNormal
import numpy as np
from tensorflow.keras.models import Model
import importlib
import tensorflow as tf
import layers
import pandas as pd
importlib.reload(layers)
from feature import *
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense, Reshape, GlobalAveragePooling1D, Dropout, BatchNormalization, Add
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, Dense, Input, Concatenate, Flatten, Layer


class DINAttention(layers.Layer):
    def __init__(self, hidden_units=[80, 40], **kwargs):
        super(DINAttention, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dense_layers = [layers.Dense(units) for units in hidden_units]
        self.dice_layers = [Dice() for _ in hidden_units]
        # 缺少最终输出层！
        # self.output_layer = layers.Dense(1, activation=None)  

    def call(self, queries, keys, mask):
        # Step 1: 扩展 queries 维度
        queries = tf.tile(queries, [1, tf.shape(keys)[1], 1])  # (None, 370, 128)
        
        # Step 2: 拼接特征
        din_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)  # (None, 370, 512)
        
        # Step 3: MLP + Dice
        for dense, dice in zip(self.dense_layers, self.dice_layers):
            din_input = dense(din_input)  # 若 hidden_units 是 [80,40]，维度变为 (None,370,80) → (None,370,40)
            din_input = dice(din_input)
        
        # Step 4: 输出注意力得分（原代码漏了这一步！）
        attention_scores = layers.Dense(1, activation=None)(din_input)  # (None, 370, 1)
        attention_scores = tf.squeeze(attention_scores, axis=-1)  # (None, 370)
        
        # Step 5: Mask 处理
        paddings = tf.ones_like(attention_scores) * (-2 ** 32 + 1)
        attention_scores = tf.where(mask, attention_scores, paddings)
        
        # Step 6: Softmax + 加权求和
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # (None, 370)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)  # (None, 370, 1)
        output = tf.reduce_sum(keys * attention_weights, axis=1)  # (None, 128)
        
        return output
    
    
class MMOELayer(Layer):
    def __init__(self, num_experts=4, num_tasks=3, units=64, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.units = units
        self.experts = [Dense(units, activation='relu') for _ in range(num_experts)]
        self.gates = [Dense(num_experts, activation='softmax') for _ in range(num_tasks)]

    def call(self, inputs):
        expert_outputs = [expert(inputs) for expert in self.experts]
        expert_stack = tf.stack(expert_outputs, axis=1)  # [B, E, U]

        task_outputs = []
        for gate in self.gates:
            gate_weights = gate(inputs)  # [B, E]
            gate_weights = tf.expand_dims(gate_weights, -1)  # [B, E, 1]
            weighted_experts = tf.reduce_sum(expert_stack * gate_weights, axis=1)  # [B, U]
            task_outputs.append(weighted_experts)
        return task_outputs
    
def get_model(model_name, feature_config, tf_config, word2vec_feed_embedding, user_embeddings, author_embeddings, is_training=False):
    if model_name=='base':
        return build_base_model(feature_config, tf_config, is_training, word2vec_feed_embedding, user_embeddings, author_embeddings)
    else:
        raise NotImplementedError


def create_embedding_matrix(features_config, feature_name, df, embedding_col):
    """
    创建嵌入矩阵，从 DataFrame 中读取预训练的嵌入向量填充到矩阵中。

    :param features_config: 特征配置字典
    :param feature_name: 特征名称
    :param df: 包含预训练嵌入向量的 DataFrame
    :param embedding_col: 嵌入向量所在的列名
    :return: 嵌入矩阵
    """
    embedding_matrix = np.zeros((
        features_config['sparse'][feature_name]['vocab_size'], 
        features_config['sparse'][feature_name]['embedding_dim'] 
    ))
    for _, row in df.iterrows():
        idx = row[feature_name]
        embedding = np.array(row[embedding_col])
        embedding_matrix[idx] = embedding
    return embedding_matrix

def get_embedding_layer(features_config, input_layers, word2vec_feed_embedding, user_embeddings, author_embeddings):
    embedding_layers = {}

    sparse = features_config['sparse'].keys()
    sequence = features_config['sequence'].keys()

    for feature_name in sparse:
        if feature_name == 'feedid':
            embedding_matrix = create_embedding_matrix(features_config, feature_name, word2vec_feed_embedding, 'feed_word2vec_embedding')
        elif feature_name == 'userid':
            embedding_matrix = create_embedding_matrix(features_config, feature_name, user_embeddings, 'user_embedding')
        elif feature_name == 'authorid':
            embedding_matrix = create_embedding_matrix(features_config, feature_name, author_embeddings, 'author_embedding')
        else:
            embedding_matrix = None

        if embedding_matrix is not None:
            initializer = tf.keras.initializers.Constant(embedding_matrix)
        else:
            initializer = tf.keras.initializers.GlorotNormal()

        embedding_layers[feature_name] = tf.keras.layers.Embedding(
            input_dim=features_config['sparse'][feature_name]['vocab_size'],
            output_dim=features_config['sparse'][feature_name]['embedding_dim'],
            input_length=1,
            embeddings_initializer=initializer,
            name=f'{feature_name}_embedding'
        )

    for feature_name in sequence:
        embedding_layers[feature_name] = tf.keras.layers.Embedding(
            input_dim=features_config['sequence'][feature_name]['vocab_size'],
            output_dim=features_config['sequence'][feature_name]['embedding_dim'],
            input_length=features_config['sequence'][feature_name]['max_len'],
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
            name=f'{feature_name}_embedding'
        )

    return embedding_layers


def build_base_model(features_config, feature_names, dense, sparse, sequence,
                word2vec_feed_embedding, 
                user_embeddings, 
                author_embeddings):
    # ======== 定义输入层 =============
    input_layers = {}
    for feature_name in feature_names:
        if feature_name in dense:
            input_layers[feature_name] = Input(shape=(1,), name=feature_name, dtype=tf.int32)
        elif feature_name in sparse:
            input_layers[feature_name] = Input(shape=(1,), name=feature_name, dtype=tf.int32)
        elif feature_name in sequence:
            input_layers[feature_name] = Input(shape=(features_config['sequence'][feature_name]['max_len'],), name=feature_name, dtype=tf.int32)
        else:
            raise ValueError(f"Unknown feature type for {feature_name}")

    embedding_layers = get_embedding_layer(features_config, input_layers, word2vec_feed_embedding, user_embeddings, author_embeddings)


    # ======== Feed Encoder =========
    # 1. 基础 Feed 特征编码（feedid/authorid 等）
    base_feed_emb_list = [
        embedding_layers['feedid'](input_layers['feedid']),
        embedding_layers['authorid'](input_layers['authorid']),
        embedding_layers['bgm_song_id'](input_layers['bgm_song_id']),
        embedding_layers['bgm_singer_id'](input_layers['bgm_singer_id']),
    ]
    base_feed_embed = layers.Concatenate(axis=-1, name='base_feed_embedding')(base_feed_emb_list)
    base_feed_embed = layers.Dense(128, activation='relu', name='base_feed_encoder')(base_feed_embed)

    # 2. Feed 侧序列特征编码
    feed_seq_feats = ['description', 'ocr', 'asr', 'manual_keyword_list', 'machine_keyword_list']
    feed_seq_emb_list = [
        embedding_layers[feat](input_layers[feat]) for feat in feed_seq_feats if feat in sequence
    ]
    # 序列特征池化
    feed_seq_embed = layers.AveragePooling1D(pool_size=2, name='feed_seq_pool')(
        layers.Concatenate(axis=-1)(feed_seq_emb_list)
    )
    feed_seq_embed = layers.Flatten()(feed_seq_embed)
    feed_seq_embed = layers.Dense(64, activation='relu', name='feed_seq_encoder')(feed_seq_embed)

    # 3. 合并 Feed 侧所有编码
    feed_embed = layers.Concatenate(axis=-1, name='feed_embed')([
        base_feed_embed, feed_seq_embed
    ])

    # ======== 用户多行为历史编码（对应 UserHistoryEncoders） =========
    # 为每种行为历史单独编码（示例：read_comment、like、follow 历史）
    history_feats = [
        'click_avatar_target_behavior_feed',
        'forward_target_behavior_feed',
        'like_target_behavior_feed',
        'read_comment_target_behavior_feed'
    ]
    history_encodings = []
    user_query = embedding_layers['userid'](input_layers['userid'])  # (None, 1, 128)
    for feat in history_feats:
        if feat in sequence:
            history_seq = embedding_layers[feat](input_layers[feat])  # (None, seq_len, dim)
            history_mask = tf.cast(input_layers[feat] > 0, tf.bool)     # (None, seq_len)
            din_vec = DINAttention(hidden_units=[80, 40])(user_query, history_seq, history_mask)
            history_encodings.append(din_vec)

    # 多行为历史拼接
    user_history_embed = layers.Concatenate(axis=-1, name='user_history_embed')(history_encodings)
    user_history_embed = layers.Dense(128, activation='relu', name='user_history_encoder')(user_history_embed)

    # ======== DLRM 低阶交互（Cosine 分支，对应架构图） =========
    # 选择参与低阶交互的特征（feed_embed + user_query + 部分 dense）
    dlrm_inputs = [
        feed_embed, 
        tf.squeeze(user_query, axis=1),  # 展平为 (None, 128)
        layers.Dense(128, activation='relu')(input_layers['videoplayseconds_bin_encoded'])  # 示例 dense 特征
    ]
    dlrm_concat = layers.Concatenate(axis=-1, name='dlrm_concat')(dlrm_inputs)

    # Batch Norm + L2 Norm + Dot Product
    dlrm_bn = layers.BatchNormalization(name='dlrm_batch_norm')(dlrm_concat)
    dlrm_l2 = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1), name='dlrm_l2_norm')(dlrm_bn)
    dlrm_dot = layers.Dense(64, activation='relu', name='dlrm_dot_product')(dlrm_l2)

    # ======== dense 特征完整编码 =========
    all_dense_inputs = [input_layers[feat] for feat in dense]
    dense_concat = layers.Concatenate(axis=-1, name='dense_concat')(all_dense_inputs)
    dense_output = layers.Dense(128, activation='relu', name='dense_encoder')(dense_concat)

    # ======== 全局特征拼接 =========
    global_features = layers.Concatenate(axis=-1, name='global_concat')([
        dlrm_dot,          # DLRM 低阶交互结果
        user_history_embed,# 用户多行为历史编码
        feed_embed,        # Feed 侧编码
        dense_output       # 全量 dense 编码
    ])

    # ======== Multiple Dropout（对应架构图） =========
    global_features = layers.Dropout(0.2, name='multiple_dropout')(global_features)

    # ======== MMOE 多任务结构（细化专家、任务分支） =========
    mmoe_outputs = MMOELayer(
        num_experts=4, 
        num_tasks=4, 
        expert_units=64
    )(global_features)

    # ======== 各任务独立塔（对应架构图多任务输出） =========
    task_outputs = []
    task_names = ['click_avatar', 'forward', 'like', 'read_comment']
    for i, task in enumerate(task_names):
        # 任务塔可叠加多层，模拟架构图 Linear/Activation
        tower = layers.Dense(64, activation='relu', name=f'{task}_tower_1')(mmoe_outputs[i])
        tower = layers.Dense(32, activation='relu', name=f'{task}_tower_2')(tower)
        # 输出 Sigmoid
        logit = layers.Dense(1, activation='sigmoid', name=f"{task}_output")(tower)
        task_outputs.append(logit)

    # ======== 模型封装 =========
    model = Model(inputs=input_layers, outputs=task_outputs)
    return model



# def build_base_model(features_config, tf_config, word2vec_feed_embedding, user_embeddings, author_embeddings, is_training=True):
#     # ============= 定义输入层 ================
#     input_layers = {}
#     for feature_name in feature_names:
#         if feature_name in dense:
#             input_layers[feature_name] = Input(shape=(1,), name=feature_name)
#         elif feature_name in sparse:
#             input_layers[feature_name] = Input(shape=(1,), name=feature_name, dtype=tf.int32)
#         elif feature_name in sequence:
#             input_layers[feature_name] = Input(shape=(features_config['sequence'][feature_name]['max_len'],), name=feature_name, dtype=tf.int32)
#         else:
#             raise ValueError(f"Unknown feature type for {feature_name}")
    
#     # ========== 构建embedding层 ================
#     embedding_layers = get_embedding_layer(features_config, tf_config, input_layers, word2vec_feed_embedding, user_embeddings, author_embeddings)
    
#     # ========== 构建模型 ===============
#     # ------ Feed Encoder ------
#     feed_embed = tf.concat([
#         embedding_layers['feedid'](input_layers['feedid']),
#         embedding_layers['authorid'](input_layers['authorid']),
#         embedding_layers['bgm_song_id'](input_layers['bgm_song_id']),
#         embedding_layers['bgm_singer_id'](input_layers['bgm_singer_id']),
#     ], axis=-1)

#     # --- 用户历史序列编码（以 read_comment_target_behavior_feed 为例） ---
#     history_seq = embedding_layers['read_comment_target_behavior_feed'](input_layers['read_comment_target_behavior_feed'])  # (None, 370, 128),
#     history_mask = tf.cast(input_layers['read_comment_target_behavior_feed'] > 0, tf.bool) # (None, 370)
#     user_query = embedding_layers['userid'](input_layers['userid']) # (None, 1, 128)

#     din_vec = DINAttention()(user_query, history_seq, history_mask)

#     # --- dense 特征拼接 ---
#     dense_inputs = tf.concat([input_layers[k] for k in features_config['dense']], axis=-1)
#     dense_output = Dense(128, activation='relu')(dense_inputs)

#     # 扩展 din_vec 和 dense_output 的维度，使其与 feed_embed 和 user_query 维度匹配
#     din_vec_expanded = tf.expand_dims(din_vec, axis=1)  # 形状从 (None, 128) 变为 (None, 1, 128)
#     dense_output_expanded = tf.expand_dims(dense_output, axis=1)  # 形状从 (None, 128) 变为 (None, 1, 128)

#     # --- 拼接所有特征 ---
#     concat_vec = Concatenate()([feed_embed, din_vec_expanded, user_query, dense_output_expanded])
#     # concat_vec = Concatenate()([feed_embed, din_vec, user_query, dense_output])

#     # --- MMOE 多任务结构 ---
#     mmoe_outputs = MMOELayer(num_experts=4, num_tasks=3)(concat_vec)

#     # --- 各任务独立塔 ---
#     task_outputs = []
#     task_names = ['click_avatar', 'forward', 'like', 'read_comment']
#     for i, task in enumerate(task_names):
#         tower = Dense(64, activation='relu')(mmoe_outputs[i])
#         logit = Dense(1, activation='sigmoid', name=f"{task}_output_{i}")(tower)
#         task_outputs.append(logit)

#     model = models.Model(inputs=input_layers, outputs=task_outputs)
#     return model


# def get_embedding_layer(features_config, tf_config, input_layers, word2vec_feed_embedding, user_embeddings, author_embeddings):
#     embedding_layers = {}

#     for feature_name in sparse:
#         # feedid使用基于word2vec得到的embedding做初始化
#         if feature_name=='feedid':
#             embedding_matrix = np.zeros((
#                 features_config['sparse']['feedid']['vocab_size'], 
#                 features_config['sparse']['feedid']['embedding_dim'] 
#             ))
#             # 填充embedding矩阵 
#             for _, row in word2vec_feed_embedding.iterrows():
#                 feedid = row['feedid']
#                 embedding = np.array(row['feed_word2vec_embedding'])
#                 embedding_matrix[feedid] = embedding
            
#             embedding_layers[feature_name] = Embedding(
#                 input_dim=features_config['sparse'][feature_name]['vocab_size'],
#                 output_dim = features_config['sparse'][feature_name]['embedding_dim'],
#                 input_length = 1,
#                 embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
#                 name='feedid_embedding'
#             )(input_layers['feedid'])
        
#         elif feature_name=='userid':
#             embedding_matrix = np.zeros((
#                 features_config['sparse']['userid']['vocab_size'], 
#                 features_config['sparse']['userid']['embedding_dim']
#             ))
#             # 填充embedding矩阵 
#             for _, row in user_embeddings.iterrows():
#                 userid = row['userid']
#                 embedding = np.array(row['user_embedding'])
#                 embedding_matrix[userid] = embedding
        
#             embedding_layers[feature_name] = Embedding(
#                 input_dim=features_config['sparse'][feature_name]['vocab_size'],
#                 output_dim = features_config['sparse'][feature_name]['embedding_dim'],
#                 input_length = 1,
#                 embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
#                 name='userid_embedding'
#             )(input_layers['userid'])
        
#         elif feature_name=='authorid':
#             embedding_matrix = np.zeros([
#                 features_config['sparse']['authorid']['vocab_size'],
#                 features_config['sparse']['authorid']['embedding_dim']
#             ])
#             for _, row in author_embeddings.iterrows():
#                 authorid = row['authorid']
#                 embedding = np.array(row['author_embedding'])
#                 embedding_matrix[authorid] = embedding
            
#             embedding_layers[feature_name] = Embedding(
#                 input_dim = features_config['sparse'][feature_name]['vocab_size'],
#                 output_dim = features_config['sparse'][feature_name]['embedding_dim'],
#                 input_length = 1,
#                 embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
#                 name='authorid_embedding'
#             )(input_layers['authorid'])
#         else:
#             embedding_layers[feature_name] = Embedding(
#                 input_dim=features_config['sparse'][feature_name]['vocab_size'],
#                 output_dim=features_config['sparse'][feature_name]['embedding_dim'],
#                 input_length=1,
#                 embeddings_initializer = tf.keras.initializers.GlorotNormal(), # 使用Deepwalk生成的embedding进行初始化
#                 name=f'{feature_name}_embedding'
#             )(input_layers[feature_name])

#     # 2. 对sequence feature做embedding层
#     for feature_name in sequence:
#         embedding_layers[feature_name] = Embedding(
#             input_dim=features_config['sequence'][feature_name]['vocab_size'],
#             output_dim=features_config['sequence'][feature_name]['embedding_dim'],
#             input_length=features_config['sequence'][feature_name]['max_len'],
#             embeddings_initializer = tf.keras.initializers.GlorotNormal(), # 正态分布初始化
#             name=f'{feature_name}_embedding'
#         )(input_layers[feature_name])

#     return embedding_layers
