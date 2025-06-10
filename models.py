from tensorflow.keras.layers import *
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import TruncatedNormal
import numpy as np
from tensorflow.keras.models import Model
from model_config import dropout_rate, stddev, num_experts, expert_units
from layers import _apply_attention
import importlib
import layers
importlib.reload(layers)
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense, Reshape, GlobalAveragePooling1D, Dropout


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
        # 保持形状一致（适配后续拼接）
        emb = Reshape((1,))(inputs[feat])  
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
        # 处理 vocab_size
        vocab_size = embedding_feat_dict['sequence'][feat]['vocab_size']
        emb_dim = embedding_feat_dict['sequence'][feat]['embedding_dim']
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
        varlen_embeddings.append(emb)
    
    return dense_embeddings, sparse_embeddings, varlen_embeddings


def build_expert_network(concat_features, num_experts=num_experts, expert_units=expert_units):
    """
    构建 MMoE 的专家网络
    :param concat_features: 拼接后的全局特征（来自 process_features 拼接结果）
    :param num_experts: 专家数量（可调整）
    :param expert_units: 每个专家隐藏层单元数（可调整）
    :return: experts 列表，每个元素是专家网络输出
    """
    experts = []
    for i in range(num_experts):
        # 专家网络第一层
        expert = Dense(expert_units, activation='relu', 
                      kernel_initializer=TruncatedNormal(stddev=stddev),
                      name=f'expert_{i}')(concat_features)
        # 专家网络第二层（加深网络）
        expert = Dense(expert_units, activation='relu',
                      kernel_initializer=TruncatedNormal(stddev=stddev),
                      name=f'expert_{i}_2')(expert)
        experts.append(expert)
    
    return experts


def build_task_networks(concat_features, experts, task_names, num_experts=4):
    """
    为每个任务构建门控 + 任务塔
    :param concat_features: 全局拼接特征
    :param experts: 专家网络输出列表（来自 build_expert_network）
    :param task_names: 任务名称列表，如 ['click', 'conversion']
    :param num_experts: 专家数量（需与 build_expert_network 一致）
    :return: task_outputs 列表，每个元素是任务输出
    """
    task_outputs = []
    for task_name in task_names:
        # 任务门控（softmax 输出专家权重）
        gate = Dense(num_experts, activation='softmax', 
                     name=f'gate_{task_name}')(concat_features)
        
        # 加权融合专家输出
        weighted_experts = []
        for i in range(num_experts):
            # Lambda 层实现：专家输出 * 门控权重
            weighted_expert = Lambda(
                lambda x: x[0] * x[1][:, i:i+1], 
                name=f'weighted_expert_{task_name}_{i}'
            )([experts[i], gate])
            weighted_experts.append(weighted_expert)
        
        # 合并加权后的专家特征
        task_input = Add()(weighted_experts)
        
        # 任务特定塔（两层全连接 + Dropout）
        tower = Dense(32, activation='relu',
                     kernel_initializer=TruncatedNormal(stddev=stddev),
                     name=f'tower_{task_name}_1')(task_input)
        tower = Dropout(dropout_rate)(tower)
        tower = Dense(16, activation='relu',
                     kernel_initializer=TruncatedNormal(stddev=stddev),
                     name=f'tower_{task_name}_2')(tower)
        
        # 任务输出（二分类用 sigmoid）
        task_output = Dense(1, activation='sigmoid',
                           name=task_name)(tower)
        task_outputs.append(task_output)
    
    return task_outputs


def MMoE_model(dense_features, sparse_features, varlen_features, task_names, embedding_feat_dict):
    """
    完整 MMoE 模型构建主函数（整合各模块）
    :param dense_features: 稠密特征列表
    :param sparse_features: 稀疏特征列表
    :param varlen_features: 变长序列特征列表
    :param encoder: 特征编码字典
    :param task_names: 任务名称列表
    :param embedding_feat_dict: 嵌入特征配置字典（给变长特征用）
    :return: 完整 Keras 模型
    """
    # 1. 构建输入层
    inputs = build_input_layers(
        dense_features, 
        sparse_features, 
        varlen_features, 
        embedding_feat_dict
    )
    
    # 2. 处理特征并拼接
    dense_emb, sparse_emb, varlen_emb = process_features(
        inputs, 
        dense_features, 
        sparse_features, 
        varlen_features, 
        embedding_feat_dict,
        use_sequence_model = 'gru_attn'
    )
    concat_features = Concatenate()(dense_emb + sparse_emb + varlen_emb)
    
    # 3. 构建专家网络
    experts = build_expert_network(concat_features)
    
    # 4. 构建任务网络
    task_outputs = build_task_networks(concat_features, experts, task_names)
    
    # 5. 组装模型
    model = Model(inputs=list(inputs.values()), outputs=task_outputs)
    
    return model


# def MMoE_model(dense_features, sparse_features, varlen_features, encoder, task_names):
#     # 输入层定义
#     inputs = {}
    
#     # 稠密特征输入
#     for feat in dense_features:
#         inputs[feat] = Input(shape=(1,), name=feat)
    
#     # 稀疏特征输入
#     for feat in sparse_features:
#         inputs[feat] = Input(shape=(1,), name=feat)
    
#     # 变长序列特征输入
#     for feat in varlen_features:
#         max_len=embedding_feat_dict['sequence'][feat]['vocab_size']-1
#         inputs[feat] = Input(shape=(max_len,), name=feat)
    
#     # 特征处理
#     ## 稠密特征处理
#     dense_embeddings = []
#     for feat in dense_features:
#         emb = Reshape((1,))(inputs[feat])  # 保持形状一致
#         dense_embeddings.append(emb)
    
#     ## 稀疏特征嵌入
#     sparse_embeddings = []
#     for feat in sparse_features:
#         vocab_size = len(encoder[feat]) + 1
#         emb_dim = min(6, int(np.sqrt(vocab_size)))  # 自适应嵌入维度
#         emb = Embedding(vocab_size, emb_dim, name=f'emb_{feat}')(inputs[feat])
#         emb = Reshape((emb_dim,))(emb)
#         sparse_embeddings.append(emb)
    
#     ## 变长序列特征处理
#     varlen_embeddings = []
#     for feat in varlen_features:
#         vocab_size = len(encoder[feat]) + 1 if feat in encoder else 100
#         emb_dim = min(6, int(np.sqrt(vocab_size)))
#         emb = Embedding(vocab_size, emb_dim, name=f'emb_{feat}')(inputs[feat])
#         emb = GlobalAveragePooling1D()(emb)  # 对序列做平均池化
#         varlen_embeddings.append(emb)
    
#     # 拼接所有特征
#     concat_features = Concatenate()(dense_embeddings + sparse_embeddings + varlen_embeddings)
    
#     # 专家网络参数
#     num_experts = 4  # 专家数量
#     expert_units = 64  # 每个专家的隐藏单元数
#     experts = []
    
#     # 创建专家网络
#     for i in range(num_experts):
#         expert = Dense(expert_units, activation='relu', 
#                       kernel_initializer=TruncatedNormal(stddev=0.02),
#                       name=f'expert_{i}')(concat_features)
#         expert = Dense(expert_units, activation='relu',
#                       kernel_initializer=TruncatedNormal(stddev=0.02),
#                       name=f'expert_{i}_2')(expert)
#         experts.append(expert)
    
#     # 任务特定参数
#     task_outputs = []
#     for task_name in task_names:
#         # 创建任务特定的门控网络
#         gate = Dense(num_experts, activation='softmax', 
#                      name=f'gate_{task_name}')(concat_features)
        
#         # 加权专家输出
#         weighted_experts = []
#         for i in range(num_experts):
#             weighted_expert = Lambda(lambda x: x[0] * x[1][:, i:i+1], 
#                                    name=f'weighted_expert_{task_name}_{i}')([experts[i], gate])
#             weighted_experts.append(weighted_expert)
        
#         # 合并加权专家
#         task_input = Add()(weighted_experts)
        
#         # 任务特定塔网络
#         tower = Dense(32, activation='relu',
#                      kernel_initializer=TruncatedNormal(stddev=0.02),
#                      name=f'tower_{task_name}_1')(task_input)
#         tower = Dropout(0.2)(tower)
#         tower = Dense(16, activation='relu',
#                      kernel_initializer=TruncatedNormal(stddev=0.02),
#                      name=f'tower_{task_name}_2')(tower)
        
#         # 任务输出层
#         task_output = Dense(1, activation='sigmoid',
#                            name=task_name)(tower)
#         task_outputs.append(task_output)
    
#     # 创建模型
#     model = Model(inputs=list(inputs.values()), outputs=task_outputs)
    
#     return model