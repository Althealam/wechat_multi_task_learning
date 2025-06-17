import networkx as nx
import numpy as np
import pandas as pd
import os
from karateclub import DeepWalk
import multiprocessing
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Activation


def generate_deepwalk_embedding(data, video_features, embed_dim=128, walks_per_node=10, walk_length=80):
    print("======== 开始生成DeepWalk Embedding==========")

    print("原始的用户视频交互数量为:", len(data))
    print("原始的视频数量为:", len(video_features))
    num_cores = multiprocessing.cpu_count()
    print("cpu数量:", num_cores)

    # 去掉空值
    data = data.dropna(subset=['userid', 'feedid'])
    video_features = video_features.dropna(subset=['feedid'])
    print("去掉空值后的用户视频交互数量为:", len(data))
    print("去掉空值后的视频数量为:", len(video_features))

    # 去重
    # data = data.drop_duplicates(['user_id', 'feedid'])
    video_features = video_features.drop_duplicates(['feedid'])
    print("去掉重复值后的视频数量为:",len(video_features))

    # 取出所有出现在交互中的feed
    feed_in_data = data['feedid'].unique()

    # 只保留 data 中有交互记录的 feed
    video_features = video_features[video_features['feedid'].isin(feed_in_data)]
    print("有交互历史的feed数量为:", len(set(video_features['feedid'])))
    # 重新编号 userid 和 feedid，确保编号为 0-N 的连续整数
    all_nodes = pd.Series(np.concatenate([data['userid'].unique(), data['feedid'].unique()]))
    node2id = {node: idx for idx, node in enumerate(all_nodes)} # 新图节点id->原node2id的id
    id2node = {idx: node for node, idx in node2id.items()} # 原node2id的id->新图节点id

    # 构建图
    G = nx.Graph()
    user_feed_edges = data[['userid', 'feedid']].values
    mapped_edges = [(node2id[user], node2id[feed]) for user, feed in user_feed_edges]
    G.add_edges_from(mapped_edges)

    # 转换为连续整数节点索引
    G = nx.convert_node_labels_to_integers(G, label_attribute="orig_id")
    mapping = nx.get_node_attributes(G, "orig_id") # 新ID->原始连续整数
    inv_mapping = {v: k for k, v in mapping.items()}  # 原始连续整数->新ID

    # 训练 DeepWalk
    print("图节点数量:", G.number_of_nodes())
    print("图边数量:", G.number_of_edges())
    print("开始训练DeepWalk")
    # DeepWalk是黑盒，不会展示进度
    model = DeepWalk(
        walk_number=walks_per_node,
        walk_length=walk_length,
        dimensions=embed_dim,
        workers=num_cores
    )
    model.fit(G)
    embeddings = model.get_embedding()
    print("模型训练结束")

    # 提取 feed 节点的 embedding
    feed_embedding_dict = {}  # 使用字典存储每个feedid对应的embedding

    for feed in video_features['feedid'].unique():
        if feed in node2id:
            node_idx = node2id[feed] # 原始feed_id->原始连续整数 
            new_node_idx = inv_mapping[node_idx] # 原始连续整数->新图ID
            # 将每个feedid的embedding作为一个numpy数组存储
            feed_embedding_dict[feed] = embeddings[new_node_idx]

    # 创建DataFrame，将整个embedding向量存入一个列
    embed_df = pd.DataFrame({
        'feedid': list(feed_embedding_dict.keys()),
        'deepwalk_embedding': list(feed_embedding_dict.values())
    })


    # 保存
    os.makedirs('./data/embeddings', exist_ok=True)
    embed_df.to_csv('./data/embeddings/deepwalk_feed_embedding.csv', index=False)

    print("DeepWalk Embedding生成完成")
    return embed_df


def str_to_array(emb_str):
    """将字符串格式的embedding转换为numpy数组"""
    if isinstance(emb_str, str):
        # 处理类似 "[0.1 0.2 0.3]" 的格式
        return np.array([float(x) for x in emb_str.strip('[]').split()])
    return emb_str  # 如果已经是数组，直接返回

def align_features(deepwalk_emb, multimodal_emb, target_dim=None):
    """将两种embedding特征对齐到相同维度"""
    deepwalk_dim = len(deepwalk_emb)
    multimodal_dim = len(multimodal_emb)
    
    # 如果未指定目标维度，默认使用较大的维度
    if target_dim is None:
        target_dim = max(deepwalk_dim, multimodal_dim)
    
    # 对低维特征进行升维（通过全连接层）
    if deepwalk_dim < target_dim:
        # 创建简单的映射模型
        input_layer = Input(shape=(deepwalk_dim,))
        output_layer = Dense(target_dim, activation='relu')(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        deepwalk_emb = model.predict(np.array([deepwalk_emb]))[0]
    
    if multimodal_dim < target_dim:
        input_layer = Input(shape=(multimodal_dim,))
        output_layer = Dense(target_dim, activation='relu')(input_layer)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        multimodal_emb = model.predict(np.array([multimodal_emb]))[0]
    
    return deepwalk_emb, multimodal_emb

def feature_concat_fusion(deepwalk_feed_embedding, feed_embeddings):
    """特征对齐后拼接融合"""
    # 转换字符串格式的embedding
    if isinstance(deepwalk_feed_embedding, str):
        deepwalk_feed_embedding = str_to_array(deepwalk_feed_embedding)
    
    if isinstance(feed_embeddings, str):
        feed_embeddings = str_to_array(feed_embeddings)
    
    # 特征对齐
    deepwalk_aligned, multimodal_aligned = align_features(
        deepwalk_feed_embedding, 
        feed_embeddings
    )
    
    # 拼接特征
    return np.concatenate([deepwalk_aligned, multimodal_aligned])

def attention_weighted_fusion(deepwalk_feed_embedding, feed_embeddings):
    """注意力加权融合
    deepwalk_feed_embedding: (4377, 2)
    feed_embeddings: (3312, 2)
    """
        # 调试：打印输入数据的类型和形状
    # print(f"DeepWalk输入类型: {type(deepwalk_feed_embedding)}")
    # if hasattr(deepwalk_feed_embedding, 'shape'):
    #     print(f"DeepWalk输入形状: {deepwalk_feed_embedding.shape}")
    # else:
    #     print(f"DeepWalk输入长度: {len(deepwalk_feed_embedding)}")
    
    # print(f"多模态输入类型: {type(feed_embeddings)}")
    # if hasattr(feed_embeddings, 'shape'):
    #     print(f"多模态输入形状: {feed_embeddings.shape}")
    # else:
    #     print(f"多模态输入长度: {len(feed_embeddings)}")
    
    # 转换字符串格式的embedding（如果需要）
    if isinstance(deepwalk_feed_embedding, str):
        deepwalk_feed_embedding = str_to_array(deepwalk_feed_embedding)
    
    if isinstance(feed_embeddings, str):
        feed_embeddings = str_to_array(feed_embeddings)
    
    
    # 转换字符串格式的embedding
    if isinstance(deepwalk_feed_embedding, str):
        deepwalk_feed_embedding = str_to_array(deepwalk_feed_embedding)
    
    if isinstance(feed_embeddings, str):
        feed_embeddings = str_to_array(feed_embeddings)
    
    # 特征对齐
    target_dim = max(len(deepwalk_feed_embedding), len(feed_embeddings))
    deepwalk_aligned, multimodal_aligned = align_features(
        deepwalk_feed_embedding, 
        feed_embeddings,
        target_dim=target_dim
    )
    
    # 创建注意力融合模型
    deepwalk_input = Input(shape=(target_dim,))
    multimodal_input = Input(shape=(target_dim,))
    
    # 计算注意力权重
    concat_input = Concatenate()([deepwalk_input, multimodal_input])
    attention_score = Dense(1, activation='sigmoid')(concat_input)
    
    # 加权融合
    weighted_deepwalk = tf.multiply(deepwalk_input, attention_score)
    weighted_multimodal = tf.multiply(multimodal_input, 1 - attention_score)
    fused_output = weighted_deepwalk + weighted_multimodal
    
    model = tf.keras.Model(
        inputs=[deepwalk_input, multimodal_input],
        outputs=fused_output
    )
    
    # 将输入转换为NumPy数组并调整形状为(batch_size=1, embedding_dim)
    deepwalk_array = np.array([deepwalk_aligned]).astype(np.float32)
    multimodal_array = np.array([multimodal_aligned]).astype(np.float32)
    
    # 预测融合结果
    fused_embedding = model.predict([deepwalk_array, multimodal_array], verbose=0)[0]
    
    return fused_embedding

def dual_branch_network_fusion(deepwalk_feed_embedding, feed_embeddings):
    """双分支网络融合"""
    # 转换字符串格式的embedding
    if isinstance(deepwalk_feed_embedding, str):
        deepwalk_feed_embedding = str_to_array(deepwalk_feed_embedding)
    
    if isinstance(feed_embeddings, str):
        feed_embeddings = str_to_array(feed_embeddings)
    
    # 特征对齐
    target_dim = max(len(deepwalk_feed_embedding), len(feed_embeddings))
    deepwalk_aligned, multimodal_aligned = align_features(
        deepwalk_feed_embedding, 
        feed_embeddings,
        target_dim=target_dim
    )
    
    # 创建双分支网络
    deepwalk_input = Input(shape=(target_dim,))
    multimodal_input = Input(shape=(target_dim,))
    
    # DeepWalk分支
    deepwalk_branch = Dense(target_dim//2, activation='relu')(deepwalk_input)
    deepwalk_branch = Dense(target_dim//4, activation='relu')(deepwalk_branch)
    
    # 多模态分支
    multimodal_branch = Dense(target_dim//2, activation='relu')(multimodal_input)
    multimodal_branch = Dense(target_dim//4, activation='relu')(multimodal_branch)
    
    # 融合分支
    merged = Concatenate()([deepwalk_branch, multimodal_branch])
    merged = Dense(target_dim//2, activation='relu')(merged)
    output = Dense(target_dim, activation='relu')(merged)
    
    model = tf.keras.Model(
        inputs=[deepwalk_input, multimodal_input],
        outputs=output
    )
    
    # 预测融合结果
    fused_embedding = model.predict([
        np.array([deepwalk_aligned]),
        np.array([multimodal_aligned])
    ])[0]
    
    return fused_embedding

def concat_feed_embedding(deepwalk_feed_embedding, feed_embeddings, mode):
    """
    融合deepwalk生成的embedding和多模态embedding
    
    参数:
        deepwalk_feed_embedding: DeepWalk生成的embedding（可以是numpy数组或字符串格式）
        feed_embeddings: 多模态embedding（可以是numpy数组或字符串格式）
        mode: 融合模式，可选值为 'concat'（特征拼接）、'attention'（注意力加权）、'dual_branch'（双分支网络）
    
    返回:
        融合后的embedding
    """
    if mode == 'concat': # 简单拼接
        return feature_concat_fusion(deepwalk_feed_embedding, feed_embeddings)
    elif mode == 'attention': # 注意力加权
        return attention_weighted_fusion(deepwalk_feed_embedding, feed_embeddings)
    elif mode == 'dual_branch': # 双分支网络
        return dual_branch_network_fusion(deepwalk_feed_embedding, feed_embeddings)
    else:
        raise ValueError("不支持的融合模式，请选择 'concat', 'attention' 或 'dual_branch'")