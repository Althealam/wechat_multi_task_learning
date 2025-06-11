from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from model_config import dropout_rate, stddev, num_experts, expert_units
import importlib
import tensorflow as tf
import layers
importlib.reload(layers)
from layers import _apply_attention, positional_encoding
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense, Reshape, GlobalAveragePooling1D, Dropout, BatchNormalization, Add
from model_config import dropout_rate, stddev, num_experts, expert_units
from tensorflow.keras.layers import Lambda


def build_input_layers(dense_features, sparse_features, varlen_features, embedding_feat_dict):
    """
    为稠密、稀疏、变长特征分别创建 Input 层
    :param dense_features: 稠密特征列表，如 ['age', 'income']
    :param sparse_features: 稀疏特征列表，如 ['gender', 'occupation']
    :param varlen_features: 变长序列特征列表，如 ['click_sequence']
    :param encoder: 特征编码字典（原代码依赖）
    :param embedding_feat_dict: 嵌入特征配置字典（原代码依赖）
    :return: inputs 字典，key 是特征名，value 是 Input 层
    """
    inputs = {}
    # 稠密特征输入
    for feat in dense_features:
        inputs[feat] = Input(shape=(1,), name=feat)
    
    # 稀疏特征输入
    for feat in sparse_features:
        inputs[feat] = Input(shape=(1,), name=feat)
    
    # 变长序列特征输入
    for feat in varlen_features:
        max_len = embedding_feat_dict['sequence'][feat]['max_len']
        inputs[feat] = Input(shape=(max_len,), name=feat)
    
    return inputs


def process_features(inputs, dense_features, sparse_features, varlen_features, embedding_feat_dict, use_sequence_model='avg_pool', rnn_units=64, dropout_rate=0.1):
    """
    处理输入特征，生成稠密、稀疏、变长特征的嵌入/处理结果
    :param inputs: build_input_layers 返回的 Input 字典
    :param dense_features: 稠密特征列表
    :param sparse_features: 稀疏特征列表
    :param varlen_features: 变长序列特征列表
    :param encoder: 特征编码字典（原代码依赖）
    :param use_sequence_model: 对变长序列特征的处理方式
    :return: 处理后的特征列表（dense_embeddings + sparse_embeddings + varlen_embeddings）
    """
    # 稠密特征处理
    dense_embeddings = []
    for feat in dense_features:
        dense_feat = inputs[feat]
        dense_feat = BatchNormalization()(dense_feat)
        # 保持形状一致（适配后续拼接）
        emb = Reshape((1,))(dense_feat)  
        dense_embeddings.append(emb)
    
    # 稀疏特征嵌入
    sparse_embeddings = []
    for feat in sparse_features:
        vocab_size = embedding_feat_dict['sparse'][feat]['vocab_size']
        emb_dim = embedding_feat_dict['sparse'][feat]['embedding_dim']
        emb = Embedding(vocab_size, emb_dim, name=f'emb_{feat}')(inputs[feat])
        # 展平为一维嵌入向量
        emb = Reshape((emb_dim,))(emb)  
        sparse_embeddings.append(emb)
    
    # 变长序列特征处理
    varlen_embeddings = []
    for feat in varlen_features:
        vocab_size = embedding_feat_dict['sequence'][feat]['vocab_size']
        emb_dim = embedding_feat_dict['sequence'][feat]['embedding_dim']
        if feat=='description': # 只对description选择lstm, gru, bi_lstm；除了description之外的特征都选择平均池化
            emb = Embedding(vocab_size, emb_dim, name=f'emb_{feat}')(inputs[feat])
            # 对变长序列做平均池化，压缩成固定长度
            if use_sequence_model=='avg_pool':
                emb = GlobalAveragePooling1D()(emb)  
            elif use_sequence_model=='lstm':
                emb = LSTM(rnn_units)(emb)
            elif use_sequence_model=='gru':
                emb = GRU(rnn_units)(emb)
            elif use_sequence_model=='bi_lstm':
                # 双向LSTM
                emb = Bidirectional(LSTM(rnn_units//2))(emb)
            elif use_sequence_model=='lstm_attn':
                lstm_out = LSTM(rnn_units, return_sequence=True, dropout=dropout_rate)(emb)
                emb = _apply_attention(lstm_out)
            elif use_sequence_model=='gru_attn':
                gru_out = GRU(rnn_units, return_sequences=True, dropout=dropout_rate)(emb)
                emb = _apply_attention(gru_out)
            else:
                raise ValueError(f"不支持的序列模型: {use_sequence_model}")
        else: # 除了description之外的特征用平均池化
            emb = Embedding(vocab_size, emb_dim, name=f'emb_{feat}')(inputs[feat])
            emb = GlobalAveragePooling1D()(emb) 
        varlen_embeddings.append(emb)
    
    return dense_embeddings, sparse_embeddings, varlen_embeddings



def gate_layer(experts, input_tensor, num_experts, name_prefix):
    gate_logits = Dense(num_experts, activation='softmax', name=f"{name_prefix}_gate_logits")(input_tensor)
    
    # Shape: (batch_size, expert_units) for each expert
    # Shape after stacking: (batch_size, num_experts, expert_units)
    expert_stack = Lambda(lambda x: tf.stack(x, axis=1), name=f"{name_prefix}_expert_stack")(experts)
    
    # Shape of gate_logits: (batch_size, num_experts)
    gate_output = Lambda(
        lambda x: tf.reduce_sum(x[0] * tf.expand_dims(x[1], axis=-1), axis=1),
        name=f"{name_prefix}_gate_output"
    )([expert_stack, gate_logits])
    
    return gate_output

def build_experts(input_tensor, num_experts, expert_units, name_prefix):
    experts = []
    for i in range(num_experts):
        expert = Dense(expert_units, activation='relu',
                       kernel_initializer=TruncatedNormal(stddev=0.01),
                       name=f'{name_prefix}_expert_{i}')(input_tensor)
        # print(f"{name_prefix}_expert_{i} output shape: {expert.shape}")
        experts.append(expert)
    return experts

def PLE_layer(input_tensor, task_names, num_shared_experts, num_task_experts, expert_units, name_prefix):
    shared_experts = build_experts(input_tensor, num_shared_experts, expert_units, name_prefix=f'{name_prefix}_shared')
    all_task_outputs = {}

    for task in task_names:
        task_experts = build_experts(input_tensor, num_task_experts, expert_units, name_prefix=f'{name_prefix}_{task}')
        # 合并共享+特定专家
        combined_experts = task_experts + shared_experts
        # print(f"combined_experts length: {len(combined_experts)}")
        # for i, e in enumerate(combined_experts):
        #     print(f"expert {i} type: {type(e)}, shape: {e.shape if hasattr(e, 'shape') else 'N/A'}")
        
        output = gate_layer(combined_experts, input_tensor, len(combined_experts), name_prefix=f'{name_prefix}_{task}')
        all_task_outputs[task] = output
        # print(output)
    return all_task_outputs

def build_PLE_model(dense_features, sparse_features, varlen_features, task_names, embedding_feat_dict,
                    num_shared_experts=2, num_task_experts=2, expert_units=64, ple_layers=2):
    # === 输入层 ===
    inputs = build_input_layers(
        dense_features, 
        sparse_features, 
        varlen_features, 
        embedding_feat_dict
    )

    # === 特征处理（此处可改为更复杂的嵌入/池化逻辑）===
    dense_emb, sparse_emb, varlen_emb = process_features(
        inputs, 
        dense_features, 
        sparse_features, 
        varlen_features, 
        embedding_feat_dict,
        use_sequence_model = 'gru_attn'
    )

    concat_input = Concatenate()(dense_emb + sparse_emb + varlen_emb)

    # === PLE 层堆叠 ===
    ple_input = concat_input
    for i in range(ple_layers):
        task_outputs = PLE_layer(ple_input, task_names, num_shared_experts, num_task_experts,
                                 expert_units, name_prefix=f'ple_layer_{i}')
        # print(task_outputs)
        # 下一层的输入是每个任务的输出拼接
        ple_input = Concatenate()(list(task_outputs.values()))

    # === 任务塔 + 输出 ===
    final_outputs = []
    for task in task_names:
        tower = Dense(32, activation='relu')(task_outputs[task])
        tower = Dropout(0.2)(tower)
        tower = Dense(16, activation='relu')(tower)
        output = Dense(1, activation='sigmoid', name=task)(tower)
        final_outputs.append(output)

    model = Model(inputs=list(inputs.values()), outputs=final_outputs)
    return model
