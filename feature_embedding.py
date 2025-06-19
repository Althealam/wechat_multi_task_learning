"""
这段代码是生成embedding的
目前包含feed的deepwalk embedding
"""
import numpy as np
import pandas as pd
import os
from karateclub import DeepWalk
import multiprocessing
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Activation
import networkx as nx
from gensim.models import Word2Vec
from tqdm import tqdm


# def generate_deepwalk_embedding(data, video_features, embed_dim=128, walks_per_node=10, walk_length=80):
#     print("======== 开始生成DeepWalk Embedding==========")

#     print("原始的user-feed交互数量为:", len(data))
#     print("原始的feed数量为:", len(video_features))
#     num_cores = multiprocessing.cpu_count()
#     print("cpu数量:", num_cores)

#     # 去掉空值
#     data = data.dropna(subset=['userid', 'feedid'])
#     video_features = video_features.dropna(subset=['feedid'])
#     print("去掉空值后的user-feed交互数量为:", len(data))
#     print("去掉空值后的feed数量为:", len(video_features))

#     # 去重
#     # data = data.drop_duplicates(['user_id', 'feedid'])
#     video_features = video_features.drop_duplicates(['feedid'])
#     print("去掉重复值后的视频数量为:",len(video_features))

#     # 取出所有出现在交互中的feed
#     feed_in_data = data['feedid'].unique()

#     # 只保留 data 中有交互记录的 feed
#     video_features = video_features[video_features['feedid'].isin(feed_in_data)]
#     print("有交互历史的feed数量为:", len(set(video_features['feedid'])))
#     # 重新编号 userid 和 feedid，确保编号为 0-N 的连续整数
#     all_nodes = pd.Series(np.concatenate([data['userid'].unique(), data['feedid'].unique()]))
#     node2id = {node: idx for idx, node in enumerate(all_nodes)} # 新图节点id->原node2id的id
#     id2node = {idx: node for node, idx in node2id.items()} # 原node2id的id->新图节点id

#     # 构建图
#     G = nx.Graph()
#     user_feed_edges = data[['userid', 'feedid']].values
#     mapped_edges = [(node2id[user], node2id[feed]) for user, feed in user_feed_edges]
#     G.add_edges_from(mapped_edges)

#     # 转换为连续整数节点索引
#     G = nx.convert_node_labels_to_integers(G, label_attribute="orig_id")
#     mapping = nx.get_node_attributes(G, "orig_id") # 新ID->原始连续整数
#     inv_mapping = {v: k for k, v in mapping.items()}  # 原始连续整数->新ID

#     # 训练 DeepWalk
#     print("图节点数量:", G.number_of_nodes())
#     print("图边数量:", G.number_of_edges())
#     print("开始训练DeepWalk")
#     # DeepWalk是黑盒，不会展示进度
#     model = DeepWalk(
#         walk_number=walks_per_node,
#         walk_length=walk_length,
#         dimensions=embed_dim,
#         workers=num_cores
#     )
#     model.fit(G)
#     embeddings = model.get_embedding()
#     print("模型训练结束")

#     # 提取 feed 节点的 embedding
#     feed_embedding_dict = {}  # 使用字典存储每个feedid对应的embedding

#     for feed in video_features['feedid'].unique():
#         if feed in node2id:
#             node_idx = node2id[feed] # 原始feed_id->原始连续整数 
#             new_node_idx = inv_mapping[node_idx] # 原始连续整数->新图ID
#             # 将每个feedid的embedding作为一个numpy数组存储
#             feed_embedding_dict[feed] = embeddings[new_node_idx]

#     # 创建DataFrame，将整个embedding向量存入一个列
#     embed_df = pd.DataFrame({
#         'feedid': list(feed_embedding_dict.keys()),
#         'deepwalk_embedding': list(feed_embedding_dict.values())
#     })

#     print("DeepWalk Embedding生成成功！")
#     return embed_df



def generate_word2vec_embedding(data, video_features, embed_dim=128, window_size=10, min_count=1, workers=4):
    print("======== 开始生成Word2Vec Embedding ==========")
    
    # 数据预处理部分保持不变
    print("原始的user-feed交互数量为:", len(data))
    print("原始的feed数量为:", len(video_features))
    
    data = data.dropna(subset=['userid', 'feedid'])
    video_features = video_features.dropna(subset=['feedid'])
    print("去掉空值后的user-feed交互数量为:", len(data))
    
    video_features = video_features.drop_duplicates(['feedid'])
    print("去掉重复值后的视频数量为:", len(video_features))
    
    feed_in_data = data['feedid'].unique()
    video_features = video_features[video_features['feedid'].isin(feed_in_data)]
    print("有交互历史的feed数量为:", len(set(video_features['feedid'])))

    # 准备训练数据 - 将每个用户的feed序列作为句子
    print("准备训练数据...")
    user_feed_sequences = data.groupby('userid')['feedid'].apply(list).values.tolist()
    
    # 添加进度条
    print("开始训练Word2Vec模型...")
    model = Word2Vec(
        sentences=tqdm(user_feed_sequences, desc="训练进度"),
        vector_size=embed_dim,
        window=window_size,
        min_count=min_count,
        workers=workers,
        epochs=10  # 可以调整迭代次数
    )
    
    # 提取embedding
    print("提取feed embedding...")
    feed_embedding_dict = {}
    for feed in tqdm(video_features['feedid'].unique(), desc="提取进度"):
        if str(feed) in model.wv:
            feed_embedding_dict[feed] = model.wv[str(feed)]
    
    # 创建DataFrame
    embed_df = pd.DataFrame({
        'feedid': list(feed_embedding_dict.keys()),
        'word2vec_embedding': list(feed_embedding_dict.values())
    })

    print("Word2Vec Embedding生成成功！")
    return embed_df