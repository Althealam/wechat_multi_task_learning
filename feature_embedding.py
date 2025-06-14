from karateclub import DeepWalk
import pandas as pd
import os
import numpy as np

def generate_deepwalk_embedding(data, video_features, embed_dim=128, walks_per_node=10, walk_length=80):
    """
    基于用户-视频交互图的共现性，使用DeepWalk生成Feed Embedding
    """
    print("开始生成DeepWalk Embedding...")
    
    # 1. 构建用户-视频交互图
    user_feed_edges = data[['userid', 'feedid']].values
    print("user_feed_edges:\n")
    print(user_feed_edges)
    unique_users = data['userid'].unique()
    unique_feeds = video_features['feedid'].unique()
    
    # 为节点分配唯一ID
    node_id = {}
    idx = 0
    for user in unique_users:
        node_id[user] = idx
        idx += 1
    for feed in unique_feeds:
        node_id[feed] = idx
        idx += 1
    
    # 构建图边
    edges = []
    for user, feed in user_feed_edges:
        edges.append((node_id[user], node_id[feed]))
    
    # 2. 使用DeepWalk生成图嵌入
    model = DeepWalk(
        walk_length=walk_length,
        number_of_walks=walks_per_node,
        dimensions=embed_dim,
        workers=8
    )
    model.fit(edges)
    embeddings = model.get_embedding()
    
    # 3. 提取视频节点的Embedding
    feed_embedding = np.zeros((len(unique_feeds), embed_dim))
    feed_node_ids = [node_id[feed] for feed in unique_feeds]
    
    for i, node_id in enumerate(feed_node_ids):
        feed_embedding[i] = embeddings[node_id]
    
    # 4. 构建Embedding DataFrame
    embed_df = pd.DataFrame(
        feed_embedding,
        columns=[f'deepwalk_emb_{i}' for i in range(embed_dim)]
    )
    embed_df['feedid'] = unique_feeds
    
    # 5. 保存Embedding
    os.makedirs('/Users/bytedance/Desktop/wechat_MTL/data/embeddings', exist_ok=True)
    embed_df.to_csv('./data/embeddings/deepwalk_feed_embedding.csv', index=False)
    
    print("DeepWalk Embedding生成完成")
    return embed_df
