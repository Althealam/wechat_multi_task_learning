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

def get_input_layers(features_config):
    input_layers = {}
    dense = features_config['dense']
    sparse = features_config['sparse'].keys()
    sequence = features_config['sequence'].keys()
    
    for feature_name in dense:
        input_layers[feature_name] = Input(shape=(1,), name=feature_name)
    
    for feature_name in sparse:
        input_layers[feature_name] = Input(shape=(1,), name=feature_name, dtype=tf.int32)
    
    for feature_name in sequence:
        max_len = features_config['sequence'][feature_name]['max_len']
        input_layers[feature_name] = Input(shape=(max_len,), name=feature_name, dtype=tf.int32)
    return input_layers

def get_embedding_layer(features_config, input_layers, word2vec_feed_embedding, user_embeddings, author_embeddings):
    embedding_layers = {}

    sparse = features_config['sparse'].keys()
    sequence = features_config['sequence'].keys()

    # 处理稀疏特征
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

    # 处理序列特征（添加平均池化）
    for feature_name in sequence:
        # 创建嵌入层
        embedding = tf.keras.layers.Embedding(
            input_dim=features_config['sequence'][feature_name]['vocab_size'],
            output_dim=features_config['sequence'][feature_name]['embedding_dim'],
            input_length=features_config['sequence'][feature_name]['max_len'],
            embeddings_initializer=tf.keras.initializers.GlorotNormal(),
            name=f'{feature_name}_embedding'
        )
        
        # 添加平均池化层
        pooling = tf.keras.layers.GlobalAveragePooling1D(name=f'{feature_name}_pooling')
        
        # 创建一个顺序模型，包含嵌入层和池化层
        embedding_layers[feature_name] = tf.keras.Sequential([
            embedding,
            pooling
        ])

    return embedding_layers


# def get_embedding_layer(features_config, input_layers, word2vec_feed_embedding, user_embeddings, author_embeddings):
#     embedding_layers = {}

#     sparse = features_config['sparse'].keys()
#     sequence = features_config['sequence'].keys()

#     for feature_name in sparse:
#         if feature_name == 'feedid':
#             embedding_matrix = create_embedding_matrix(features_config, feature_name, word2vec_feed_embedding, 'feed_word2vec_embedding')
#         elif feature_name == 'userid':
#             embedding_matrix = create_embedding_matrix(features_config, feature_name, user_embeddings, 'user_embedding')
#         elif feature_name == 'authorid':
#             embedding_matrix = create_embedding_matrix(features_config, feature_name, author_embeddings, 'author_embedding')
#         else:
#             embedding_matrix = None

#         if embedding_matrix is not None:
#             initializer = tf.keras.initializers.Constant(embedding_matrix)
#         else:
#             initializer = tf.keras.initializers.GlorotNormal()

#         embedding_layers[feature_name] = tf.keras.layers.Embedding(
#             input_dim=features_config['sparse'][feature_name]['vocab_size'],
#             output_dim=features_config['sparse'][feature_name]['embedding_dim'],
#             input_length=1,
#             embeddings_initializer=initializer,
#             name=f'{feature_name}_embedding'
#         )

#     for feature_name in sequence:
#         embedding_layers[feature_name] = tf.keras.layers.Embedding(
#             input_dim=features_config['sequence'][feature_name]['vocab_size'],
#             output_dim=features_config['sequence'][feature_name]['embedding_dim'],
#             input_length=features_config['sequence'][feature_name]['max_len'],
#             embeddings_initializer=tf.keras.initializers.GlorotNormal(),
#             name=f'{feature_name}_embedding'
#         )

#     return embedding_layers


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



# def build_model_with_pooling(features_config, word2vec_feed_embedding, user_embeddings, author_embeddings):
#     # ============== 输入层 ===============
#     input_layers = get_input_layers(features_config)
#     # ============= 嵌入层构建 ============
#     embedding_layers = get_embedding_layer(features_config, input_layers, word2vec_feed_embedding, user_embeddings, author_embeddings)

#     # 序列特征嵌入 + 池化
#     sequence_outputs = {}
#     for feature_name in sequence:
#         embedding = Embedding(
#             input_dim=features_config['sequence'][feature_name]['vocab_size'],
#             output_dim=features_config['sequence'][feature_name]['embedding_dim'],
#             input_length=features_config['sequence'][feature_name]['max_len'],
#             embeddings_initializer=tf.keras.initializers.GlorotNormal(),
#             name=f'{feature_name}_embedding'
#         )
        
#         embedded = embedding(input_layers[feature_name]) # (None, max_len, embedding_dim)
#         pooled = GlobalAveragePooling1D(name=f'pool_{feature_name}')(embedded) # (None, embedding_dim)
#         sequence_outputs[feature_name] = pooled

#     # ============= 特征拼接 ============
#     # 处理dense特征
#     dense_features = [input_layers[feature] for feature in dense]
#     dense_concatenated = Concatenate()(dense_features) if dense_features else None
    
#     # 处理sparse特征
#     sparse_features = []
#     for feature in sparse:
#         embedded = embedding_layers[feature](input_layers[feature]) # (None, 1, emb_dim)
#         # 确保嵌入输出是3D (batch_size, 1, emb_dim)
#         if len(embedded.shape) == 2:
#             embedded = tf.expand_dims(embedded, axis=1) 
#         flattened = tf.squeeze(embedded, axis=1) # (None, embed_dim)
#         sparse_features.append(flattened)
    
#     sparse_concatenated = Concatenate()(sparse_features) if sparse_features else None
    
#     # 拼接所有特征
#     all_features = []
#     all_features.append(dense_concatenated)
#     all_features.append(sparse_concatenated)
#     all_features.extend(sequence_outputs.values())
    
#     unified_features = Concatenate()(all_features)
#     # ============= 模型构建 ============
#     x = Dense(512, activation='relu')(unified_features)
#     x = Dense(256, activation='relu')(x)
    
#     outputs = {
#         'read_comment': Dense(1, activation='sigmoid', name='target_read_comment')(x),
#         'like': Dense(1, activation='sigmoid', name='target_like')(x),
#     }
    
#     return tf.keras.Model(inputs=input_layers, outputs=outputs)