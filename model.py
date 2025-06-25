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
import layers, importlib
importlib.reload(layers)
from layers import *

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

# def get_embedding_layer(features_config, input_layers, word2vec_feed_embedding, user_embeddings, author_embeddings):
#     embedding_layers = {}

#     sparse = features_config['sparse'].keys()
#     sequence = features_config['sequence'].keys()

#     # 处理稀疏特征
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

#     # 处理序列特征（添加平均池化）
#     for feature_name in sequence:
#         # 创建嵌入层
#         embedding = tf.keras.layers.Embedding(
#             input_dim=features_config['sequence'][feature_name]['vocab_size'],
#             output_dim=features_config['sequence'][feature_name]['embedding_dim'],
#             input_length=features_config['sequence'][feature_name]['max_len'],
#             embeddings_initializer=tf.keras.initializers.GlorotNormal(),
#             name=f'{feature_name}_embedding'
#         )
        
#         # 添加平均池化层
#         pooling = tf.keras.layers.GlobalAveragePooling1D(name=f'{feature_name}_pooling')
        
#         # 创建一个顺序模型，包含嵌入层和池化层
#         embedding_layers[feature_name] = tf.keras.Sequential([
#             embedding,
#             pooling
#         ])

#     return embedding_layers


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



class MMoE(Layer):
    def __init__(self, units, num_experts, num_tasks, **kwargs):
        super(MMoE, self).__init__(**kwargs)
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = [Dense(units, activation='relu') for _ in range(num_experts)]
        self.gates = [Dense(num_experts, activation='softmax') for _ in range(num_tasks)]

    def call(self, inputs):
        # 所有 experts 的输出: shape = [(batch_size, units), ...]
        expert_outputs = [expert(inputs) for expert in self.experts]  # list of (B, U)
        expert_stack = tf.stack(expert_outputs, axis=1)  # shape: (batch_size, num_experts, units)

        task_outputs = []
        for gate in self.gates:
            gate_weights = gate(inputs)  # (batch_size, num_experts)
            gate_weights = tf.expand_dims(gate_weights, axis=-1)  # (batch_size, num_experts, 1)
            weighted_expert_output = tf.reduce_sum(expert_stack * gate_weights, axis=1)  # (batch_size, units)
            task_outputs.append(weighted_expert_output)

        return task_outputs  # list of (batch_size, units)



def build_model(features_config, word2vec_feed_embedding, user_embeddings, author_embeddings):
    # ============== 输入层 ===============
    input_layers = get_input_layers(features_config)
    # ============= 嵌入层构建 ============
    embedding_layers = get_embedding_layer(features_config, input_layers, word2vec_feed_embedding, user_embeddings, author_embeddings)

    # ============= 特征拼接 ============
    # 处理sequence特征
    # 序列特征嵌入 + 池化
    sequence_outputs = {}
    sequence = features_config['sequence'].keys() if'sequence' in features_config else []
    for feature_name in sequence:
        embedding = embedding_layers[feature_name]
        embedded = embedding(input_layers[feature_name])  # (None, max_len, embedding_dim)
        sequence_outputs[feature_name] = embedded

    # 处理dense特征
    dense_outputs = {}
    for feature_name in dense:
        dense_outputs[feature_name] = input_layers[feature_name]

    # 处理sparse特征
    sparse_outputs = {}
    for feature_name in sparse:
        embedded = embedding_layers[feature_name](input_layers[feature_name]) # (None, 1, emb_dim)
        # 确保嵌入输出是3D (batch_size, 1, emb_dim)
        if len(embedded.shape) == 2:
            embedded = tf.expand_dims(embedded, axis=1) 
        flattened = tf.squeeze(embedded, axis=1) # (None, embed_dim)
        sparse_outputs[feature_name]=flattened

    # =========== 获取feedencoder, userhistoryencoder, contextencoder ===========
    # feedencoder
    feed_encoder = {}
    for feature_name in feed_features:
        if feature_name in sparse_outputs:
            feature_embedding = sparse_outputs[feature_name]
        elif feature_name in dense_outputs:
            feature_embedding = dense_outputs[feature_name]
        elif feature_name in sequence_outputs:
            feature_embedding = sequence_outputs[feature_name]
        feed_encoder[feature_name]=feature_embedding

    # userhistoryencoder
    user_history_encoder = {}
    for feature_name in user_history_features:
        if feature_name in sparse_outputs:
            feature_embedding = sparse_outputs[feature_name]
        elif feature_name in dense_outputs:
            feature_embedding = dense_outputs[feature_name]
        elif feature_name in sequence_outputs:
            feature_embedding = sequence_outputs[feature_name]
        else:
            print(f"{feature_name}不存在")
        user_history_encoder[feature_name]=feature_embedding
    
    context_encoder ={}
    for feature_name in context_features:
        if feature_name in sparse_outputs:
            feature_embedding = sparse_outputs[feature_name]
        elif feature_name in dense_outputs:
            feature_embedding = dense_outputs[feature_name]
        elif feature_name in sequence_outputs:
            feature_embedding = sequence_outputs[feature_name]
        else:
            print(f"{feature_name}不存在")
        context_encoder[feature_name]=feature_embedding

    # =============== 使用DIN层对所有历史行为向量计算embedding =============
    target_embedding = feed_encoder['feedid']  # (None, 128)

    # 获取行为序列的embedding向量
    read_comment_history_embedding = user_history_encoder['read_comment_target_behavior_feed'] # (None, 221, 128)
    like_history_embedding = user_history_encoder['like_target_behavior_feed']  # (None, 62, 128)
    click_avatar_history_embedding = user_history_encoder['click_avatar_target_behavior_feed'] # (None, 14, 128)
    forward_history_embedding = user_history_encoder['forward_target_behavior_feed'] # (None, 6, 128)
    follow_history_embedding = user_history_encoder['follow_target_behavior_feed'] # (None, 2, 128)
    favorite_history_embedding = user_history_encoder['favorite_target_behavior_feed'] # (None,2, 128)
    comment_history_embedding = user_history_encoder['comment_target_behavior_feed'] # (None, 9, 128)
    interactirve_history_embedding = user_history_encoder['interactive_history'] # (None, 229, 128)
    non_interactirve_history_embedding = user_history_encoder['non_interactive_history'] # (None, 789, 128)
    finish_history_embedding = user_history_encoder['finish_history'] # (None, 794, 128)
    unfinish_history_embedding = user_history_encoder['unfinish_history'] # (None, 1, 128)

    # 计算 attention pooling 后的兴趣表示
    din_attention_layer = DinAttention(name="din_attention")
    read_comment_interest_embedding = din_attention_layer([target_embedding, read_comment_history_embedding]) # (None, 128)
    like_interest_embedding = din_attention_layer([target_embedding, like_history_embedding]) # (None, 128)
    click_avatar_interest_embedding = din_attention_layer([target_embedding, click_avatar_history_embedding]) # (None, 128)
    forward_interest_embedding = din_attention_layer([target_embedding, forward_history_embedding]) # (None, 128)
    follow_interest_embedding = din_attention_layer([target_embedding, follow_history_embedding]) # (None, 128)
    favorite_interest_embedding = din_attention_layer([target_embedding, favorite_history_embedding]) # (None, 128)
    comment_interest_embedding = din_attention_layer([target_embedding, comment_history_embedding]) # (None, 128)
    interactive_interest_embedding = din_attention_layer([target_embedding, interactirve_history_embedding]) # (None, 128)
    non_interactirve_interest_embedding = din_attention_layer([target_embedding, non_interactirve_history_embedding])# (None, 128)
    finish_history_interest_embedding = din_attention_layer([target_embedding, finish_history_embedding])# (None, 128)
    unfinish_history_interest_embedding = din_attention_layer([target_embedding, unfinish_history_embedding])# (None, 128)

    # ============= DIN部分 =================
    # 1. L2 Normalize
    target_norm = Lambda(lambda x: l2_normalize(x), name='l2_norm_target')(target_embedding)
    read_comment_interest_embedding = Lambda(lambda x: l2_normalize(x), name='l2_norm_interest_read_comment')(read_comment_interest_embedding)
    like_interest_embedding = Lambda(lambda x: l2_normalize(x), name='l2_norm_interest_like')(like_interest_embedding)
    click_avatar_interest_embedding = Lambda(lambda x: l2_normalize(x), name='l2_norm_interest_click_avatar')(click_avatar_interest_embedding)
    forward_interest_embedding = Lambda(lambda x: l2_normalize(x), name='l2_norm_interest_forward')(forward_interest_embedding)

    follow_interest_embedding = Lambda(lambda x: l2_normalize(x), name='l2_norm_interest_follow')(follow_interest_embedding)
    favorite_interest_embedding = Lambda(lambda x: l2_normalize(x), name='l2_norm_interest_favorite')(favorite_interest_embedding)
    comment_interest_embedding = Lambda(lambda x: l2_normalize(x), name='l2_norm_interest_comment')(comment_interest_embedding)

    interactive_interest_embedding = Lambda(lambda x: l2_normalize(x), name='l2_norm_interest_interactive')(interactive_interest_embedding)
    non_interactirve_interest_embedding = Lambda(lambda x: l2_normalize(x), name='l2_norm_interest_non_interactive')(non_interactirve_interest_embedding)
    finish_history_interest_embedding = Lambda(lambda x: l2_normalize(x), name='l2_norm_interest_finish_history')(finish_history_interest_embedding)
    unfinish_history_interest_embedding = Lambda(lambda x: l2_normalize(x), name='l2_norm_interest_unfinish_history')(unfinish_history_interest_embedding)


    # 2. Dot Product (cosine)
    read_comment_cosine_similarity = Dot(axes=-1, name='cosine_similarity_read_comment')([target_norm, read_comment_interest_embedding])  # (None, 1)
    like_cosine_similarity = Dot(axes=-1, name='cosine_similarity_like')([target_norm, like_interest_embedding])  # (None, 1)
    click_avatar_cosine_similarity = Dot(axes=-1, name='cosine_similarity_click_avatar')([target_norm, click_avatar_interest_embedding])  # (None, 1)
    forward_similarity = Dot(axes=-1, name='cosine_similarity_forward')([target_norm, forward_interest_embedding])  # (None, 1)
    follow_cosine_similarity = Dot(axes=-1, name='cosine_similarity_follow')([target_norm, follow_interest_embedding])  # (None, 1)
    favorite_cosine_similarity = Dot(axes=-1, name='cosine_similarity_favorite')([target_norm, favorite_interest_embedding])  # (None, 1)
    comment_cosine_similarity = Dot(axes=-1, name='cosine_similarity_comment')([target_norm, comment_interest_embedding])  # (None, 1)
    interactive_cosine_similarity = Dot(axes=-1, name='cosine_similarity_interactive')([target_norm, interactive_interest_embedding])  # (None, 1)
    non_interactirve_cosine_similarity = Dot(axes=-1, name='cosine_similarity_non_interactive')([target_norm, non_interactirve_interest_embedding])  # (None, 1)
    finish_history_cosine_similarity = Dot(axes=-1, name='cosine_similarity_finish')([target_norm, finish_history_interest_embedding])  # (None, 1)
    unfinish_history_cosine_similarity = Dot(axes=-1, name='cosine_similarity_unfinish')([target_norm, unfinish_history_interest_embedding])  # (None, 1)

    # =========== 特征拼接 ============
    concat_features = []

    concat_features.extend([
        feed_encoder['feedid'], # 当前feed (None, 128)
        user_history_encoder['userid'], # 当前user # (None, 128)
        # 来自不同历史行为的兴趣表示
        read_comment_cosine_similarity, # (None, 1)
        like_cosine_similarity,
        click_avatar_cosine_similarity,
        forward_similarity,
        follow_cosine_similarity,
        favorite_cosine_similarity,
        comment_cosine_similarity,
        interactive_cosine_similarity,
        non_interactirve_cosine_similarity,
        finish_history_cosine_similarity,
        unfinish_history_cosine_similarity,
        # side info
        feed_encoder['authorid'],# (None, 128)
        feed_encoder['bgm_song_id'],# (None, 128)
        feed_encoder['bgm_singer_id'], # (None, 128)
        # 加入context features
        context_encoder['date_'],
        context_encoder['device'],
        context_encoder['videoplayseconds_bin_encoded'],
        context_encoder['is_complete'],
        context_encoder['follow'],
        context_encoder['favorite'],
        context_encoder['play'],
        context_encoder['stay']
    ])

    # 拼接所有特征
    final_concat = Concatenate(axis=-1, name="concat_all_features")(concat_features) # (None, 779)

    # BatchNorm 层
    x = BatchNormalization()(final_concat) # (None, 2176)

    mmoe_layer = MMoE(units=256, num_experts=8, num_tasks=4, name='mmoe_layer')
    task_specific_outputs = mmoe_layer(x) # (None, 256)

    read_comment_tower = Dense(64, activation='relu')(task_specific_outputs[0])
    read_comment_output = Dense(1, activation='sigmoid', name='read_comment_output')(read_comment_tower)

    comment_tower = Dense(64, activation='relu')(task_specific_outputs[1])
    comment_output = Dense(1, activation='sigmoid', name='comment_output')(comment_tower)

    click_avatar_tower = Dense(64, activation='relu')(task_specific_outputs[2])
    click_avatar_output = Dense(1, activation='sigmoid', name='click_avatar_output')(click_avatar_tower)

    forward_tower = Dense(64, activation='relu')(task_specific_outputs[3])
    forward_output = Dense(1, activation='sigmoid', name='forward_output')(forward_tower)

    model = Model(
        inputs=list(input_layers.values()),
        outputs=[
            read_comment_output,
            comment_output,
            click_avatar_output,
            forward_output
        ]
    )

    return model