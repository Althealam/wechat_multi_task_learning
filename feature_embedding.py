import networkx as nx
import numpy as np
import pandas as pd
import os
from karateclub import DeepWalk

def generate_deepwalk_embedding(data, video_features, embed_dim=128, walks_per_node=10, walk_length=80):
    print("开始生成DeepWalk Embedding...")

    # 去掉空值
    data = data.dropna(subset=['userid', 'feedid'])
    video_features = video_features.dropna(subset=['feedid'])

    # 取出所有出现在交互中的feed
    feed_in_data = data['feedid'].unique()
    feed_in_video = video_features['feedid'].unique()

    # 只保留 data 中有交互记录的 feed
    video_features = video_features[video_features['feedid'].isin(feed_in_data)]

    # 重新编号 userid 和 feedid，确保编号为 0-N 的连续整数
    all_nodes = pd.Series(np.concatenate([data['userid'].unique(), data['feedid'].unique()]))
    node2id = {node: idx for idx, node in enumerate(all_nodes)}
    id2node = {idx: node for node, idx in node2id.items()}

    # 构建图
    G = nx.Graph()
    user_feed_edges = data[['userid', 'feedid']].values
    mapped_edges = [(node2id[user], node2id[feed]) for user, feed in user_feed_edges]
    G.add_edges_from(mapped_edges)

    G = nx.convert_node_labels_to_integers(G)
    # 训练 DeepWalk
    print("图节点数量:", G.number_of_nodes())
    print("图边数量:", G.number_of_edges())
    print("开始训练DeepWalk")
    model = DeepWalk(
        walk_number=walks_per_node,
        walk_length=walk_length,
        dimensions=embed_dim,
        workers=8
    )
    model.fit(G)
    embeddings = model.get_embedding()
    print("模型训练结束")

    # 提取 feed 节点的 embedding
    feed_embedding = []
    feed_ids = []
    for feed in video_features['feedid'].unique():
        if feed in node2id:
            node_idx = node2id[feed]
            feed_embedding.append(embeddings[node_idx])
            feed_ids.append(feed)

    embed_df = pd.DataFrame(feed_embedding, columns=[f'deepwalk_emb_{i}' for i in range(embed_dim)])
    embed_df['feedid'] = feed_ids

    # 保存
    os.makedirs('./data/embeddings', exist_ok=True)
    embed_df.to_csv('./data/embeddings/deepwalk_feed_embedding.csv', index=False)

    print("DeepWalk Embedding生成完成")
    return embed_df
