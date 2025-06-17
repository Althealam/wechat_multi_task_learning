import networkx as nx
import numpy as np
import pandas as pd
import os
from karateclub import DeepWalk
import multiprocessing

def generate_deepwalk_embedding(data, video_features, embed_dim=128, walks_per_node=10, walk_length=80):
    print("开始生成DeepWalk Embedding...")

    num_cores = multiprocessing.cpu_count()
    print("cpu数量:", num_cores)

    # 去掉空值
    data = data.dropna(subset=['userid', 'feedid'])
    video_features = video_features.dropna(subset=['feedid'])

    # 取出所有出现在交互中的feed
    feed_in_data = data['feedid'].unique()

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

    # 转换为连续整数节点索引
    G = nx.convert_node_labels_to_integers(G, label_attribute="orig_id")
    mapping = nx.get_node_attributes(G, "orig_id")
    inv_mapping = {v: k for k, v in mapping.items()}  # 原id -> 新id

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
    feed_embedding = []
    feed_ids = []
    for feed in video_features['feedid'].unique():
        if feed in node2id:
            node_idx = node2id[feed]
            new_node_idx = inv_mapping[node_idx]
            feed_embedding.append(embeddings[new_node_idx])
            feed_ids.append(feed)

    embed_df = pd.DataFrame(feed_embedding, columns=[f'deepwalk_emb_{i}' for i in range(embed_dim)])
    embed_df['feedid'] = feed_ids

    # 保存
    os.makedirs('./data/embeddings', exist_ok=True)
    embed_df.to_csv('./data/embeddings/deepwalk_feed_embedding.csv', index=False)

    print("DeepWalk Embedding生成完成")
    return embed_df


# import torch
# from torch_geometric.nn import Node2Vec
# import networkx as nx
# import pandas as pd
# import numpy as np
# from torch_geometric.utils import from_networkx
# import os

# def generate_deepwalk_embedding_gpu(data, video_features, embed_dim=128, walks_per_node=10, walk_length=80):
#     """使用GPU版"""
#     print("开始生成DeepWalk Embedding (使用GPU)...")

#     data = data.dropna(subset=['userid', 'feedid'])
#     video_features = video_features.dropna(subset=['feedid'])

#     feed_in_data = data['feedid'].unique()
#     video_features = video_features[video_features['feedid'].isin(feed_in_data)]

#     all_nodes = pd.Series(np.concatenate([data['userid'].unique(), data['feedid'].unique()]))
#     node2id = {node: idx for idx, node in enumerate(all_nodes)}
#     id2node = {idx: node for node, idx in node2id.items()}

#     user_feed_edges = data[['userid', 'feedid']].values
#     mapped_edges = [(node2id[user], node2id[feed]) for user, feed in user_feed_edges]

#     G = nx.Graph()
#     G.add_edges_from(mapped_edges)

#     print("图节点数量:", G.number_of_nodes())
#     print("图边数量:", G.number_of_edges())

#     pyg_graph = from_networkx(G)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"使用设备: {device}")

#     model = Node2Vec(
#         edge_index=pyg_graph.edge_index,
#         embedding_dim=embed_dim,
#         walk_length=walk_length,
#         context_size=walk_length // 2,
#         walks_per_node=walks_per_node,
#         num_negative_samples=1,
#         p=1, q=1,
#         sparse=True
#     ).to(device)

#     loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
#     optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

#     model.train()
#     for epoch in range(1, 6):
#         total_loss = 0
#         for pos_rw, neg_rw in loader:
#             optimizer.zero_grad()
#             loss = model.loss(pos_rw.to(device), neg_rw.to(device))
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f'Epoch {epoch}, Loss: {total_loss:.4f}')

#     model.eval()
#     embeddings = model.embedding.weight.data.cpu().numpy()

#     feed_embedding = []
#     feed_ids = []
#     for feed in video_features['feedid'].unique():
#         if feed in node2id:
#             feed_embedding.append(embeddings[node2id[feed]])
#             feed_ids.append(feed)

#     embed_df = pd.DataFrame(feed_embedding, columns=[f'deepwalk_emb_{i}' for i in range(embed_dim)])
#     embed_df['feedid'] = feed_ids

#     os.makedirs('./data/embeddings', exist_ok=True)
#     embed_df.to_csv('./data/embeddings/deepwalk_feed_embedding.csv', index=False)

#     print("DeepWalk Embedding (GPU版) 生成完成")
#     return embed_df
