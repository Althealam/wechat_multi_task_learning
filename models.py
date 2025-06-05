from tensorflow.keras.layers import *
from tensorflow.keras.initializers import TruncatedNormal

def MMoE_model(dense_features, sparse_features, varlen_features, encoder, task_names):
    # 输入层定义
    inputs = {}
    
    # 稠密特征输入
    for feat in dense_features:
        inputs[feat] = Input(shape=(1,), name=feat)
    
    # 稀疏特征输入
    for feat in sparse_features:
        inputs[feat] = Input(shape=(1,), name=feat)
    
    # 变长序列特征输入
    for feat in varlen_features:
        max_len=embedding_feat_dict['sequence'][feat]['vocab_size']-1
        inputs[feat] = Input(shape=(max_len,), name=feat)
    
    # 特征处理
    ## 稠密特征处理
    dense_embeddings = []
    for feat in dense_features:
        emb = Reshape((1,))(inputs[feat])  # 保持形状一致
        dense_embeddings.append(emb)
    
    ## 稀疏特征嵌入
    sparse_embeddings = []
    for feat in sparse_features:
        vocab_size = len(encoder[feat]) + 1
        emb_dim = min(6, int(np.sqrt(vocab_size)))  # 自适应嵌入维度
        emb = Embedding(vocab_size, emb_dim, name=f'emb_{feat}')(inputs[feat])
        emb = Reshape((emb_dim,))(emb)
        sparse_embeddings.append(emb)
    
    ## 变长序列特征处理
    varlen_embeddings = []
    for feat in varlen_features:
        vocab_size = len(encoder[feat]) + 1 if feat in encoder else 100
        emb_dim = min(6, int(np.sqrt(vocab_size)))
        emb = Embedding(vocab_size, emb_dim, name=f'emb_{feat}')(inputs[feat])
        emb = GlobalAveragePooling1D()(emb)  # 对序列做平均池化
        varlen_embeddings.append(emb)
    
    # 拼接所有特征
    concat_features = Concatenate()(dense_embeddings + sparse_embeddings + varlen_embeddings)
    
    # 专家网络参数
    num_experts = 4  # 专家数量
    expert_units = 64  # 每个专家的隐藏单元数
    experts = []
    
    # 创建专家网络
    for i in range(num_experts):
        expert = Dense(expert_units, activation='relu', 
                      kernel_initializer=TruncatedNormal(stddev=0.02),
                      name=f'expert_{i}')(concat_features)
        expert = Dense(expert_units, activation='relu',
                      kernel_initializer=TruncatedNormal(stddev=0.02),
                      name=f'expert_{i}_2')(expert)
        experts.append(expert)
    
    # 任务特定参数
    task_outputs = []
    for task_name in task_names:
        # 创建任务特定的门控网络
        gate = Dense(num_experts, activation='softmax', 
                     name=f'gate_{task_name}')(concat_features)
        
        # 加权专家输出
        weighted_experts = []
        for i in range(num_experts):
            weighted_expert = Lambda(lambda x: x[0] * x[1][:, i:i+1], 
                                   name=f'weighted_expert_{task_name}_{i}')([experts[i], gate])
            weighted_experts.append(weighted_expert)
        
        # 合并加权专家
        task_input = Add()(weighted_experts)
        
        # 任务特定塔网络
        tower = Dense(32, activation='relu',
                     kernel_initializer=TruncatedNormal(stddev=0.02),
                     name=f'tower_{task_name}_1')(task_input)
        tower = Dropout(0.2)(tower)
        tower = Dense(16, activation='relu',
                     kernel_initializer=TruncatedNormal(stddev=0.02),
                     name=f'tower_{task_name}_2')(tower)
        
        # 任务输出层
        task_output = Dense(1, activation='sigmoid',
                           name=task_name)(tower)
        task_outputs.append(task_output)
    
    # 创建模型
    model = Model(inputs=list(inputs.values()), outputs=task_outputs)
    
    return model