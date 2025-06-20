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


def get_model(model_name, feature_config, tf_config, word2vec_feed_embedding, user_embeddings, author_embeddings, is_training=False):
    if model_name=='base':
        return build_base_model(feature_config, tf_config, is_training, word2vec_feed_embedding, user_embeddings, author_embeddings)
    else:
        raise NotImplementedError


def build_base_model(features_config, tf_config, word2vec_feed_embedding, user_embeddings, author_embeddings, is_training=True):
    # ============= 定义输入层 ================
    input_layers = {}
    for feature_name in feature_names:
        if feature_name in dense:
            input_layers[feature_name] = Input(shape=(1,), name=feature_name)
        elif feature_name in sparse:
            input_layers[feature_name] = Input(shape=(1,), name=feature_name, dtype=tf.int32)
        elif feature_name in sequence:
            input_layers[feature_name] = Input(shape=(features_config['sequence'][feature_name]['max_len'],), name=feature_name, dtype=tf.int32)
        else:
            raise ValueError(f"Unknown feature type for {feature_name}")
    
    # ========== 构建embedding层 ================
    ## TODO: 这里需要想办法传入word2vec_feed_embedding, user_embeddings, author_embeddings
    embedding_layers = get_embedding_layer(features_config, tf_config, input_layers, word2vec_feed_embedding, user_embeddings, author_embeddings)
    print("embedding_layers:", embedding_layers)


def get_embedding_layer(features_config, tf_config, input_layers, word2vec_feed_embedding, user_embeddings, author_embeddings):
    embedding_layers = {}

    for feature_name in sparse:
        # feedid使用基于word2vec得到的embedding做初始化
        if feature_name=='feedid':
            embedding_matrix = np.zeros((
                features_config['sparse']['feedid']['vocab_size'], 
                features_config['sparse']['feedid']['embedding_dim'] 
            ))
            # 填充embedding矩阵 
            for _, row in word2vec_feed_embedding.iterrows():
                feedid = row['feedid']
                embedding = np.array(row['feed_word2vec_embedding'])
                embedding_matrix[feedid] = embedding
            
            embedding_layers[feature_name] = Embedding(
                input_dim=features_config['sparse'][feature_name]['vocab_size'],
                output_dim = features_config['sparse'][feature_name]['embedding_dim'],
                input_length = 1,
                embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
                name='feedid_embedding'
            )(input_layers['feedid'])
        
        elif feature_name=='userid':
            embedding_matrix = np.zeros((
                features_config['sparse']['userid']['vocab_size'], 
                features_config['sparse']['userid']['embedding_dim']
            ))
            # 填充embedding矩阵 
            for _, row in user_embeddings.iterrows():
                userid = row['userid']
                embedding = np.array(row['user_embedding'])
                embedding_matrix[userid] = embedding
        
            embedding_layers[feature_name] = Embedding(
                input_dim=features_config['sparse'][feature_name]['vocab_size'],
                output_dim = features_config['sparse'][feature_name]['embedding_dim'],
                input_length = 1,
                embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
                name='userid_embedding'
            )(input_layers['userid'])
        
        elif feature_name=='authorid':
            embedding_matrix = np.zeros([
                features_config['sparse']['authorid']['vocab_size'],
                features_config['sparse']['authorid']['embedding_dim']
            ])
            for _, row in author_embeddings.iterrows():
                authorid = row['authorid']
                embedding = np.array(row['author_embedding'])
                embedding_matrix[authorid] = embedding
            
            embedding_layers[feature_name] = Embedding(
                input_dim = features_config['sparse'][feature_name]['vocab_size'],
                output_dim = features_config['sparse'][feature_name]['embedding_dim'],
                input_length = 1,
                embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix),
                name='authorid_embedding'
            )(input_layers['authorid'])
        else:
            embedding_layers[feature_name] = Embedding(
                input_dim=features_config['sparse'][feature_name]['vocab_size'],
                output_dim=features_config['sparse'][feature_name]['embedding_dim'],
                input_length=1,
                embeddings_initializer = tf.keras.initializers.GlorotNormal(), # 使用Deepwalk生成的embedding进行初始化
                name=f'{feature_name}_embedding'
            )(input_layers[feature_name])

    # 2. 对sequence feature做embedding层
    for feature_name in sequence:
        embedding_layers[feature_name] = Embedding(
            input_dim=features_config['sequence'][feature_name]['vocab_size'],
            output_dim=features_config['sequence'][feature_name]['embedding_dim'],
            input_length=features_config['sequence'][feature_name]['max_len'],
            embeddings_initializer = tf.keras.initializers.GlorotNormal(), # 正态分布初始化
            name=f'{feature_name}_embedding'
        )(input_layers[feature_name])

    return embedding_layers
