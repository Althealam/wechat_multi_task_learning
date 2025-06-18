"""
这段代码的作用是数据预处理和特征工程
1. 数据预处理
* 缺失值填充
* videoplayseconds分桶离散化
2. 特征工程
* 统计特征
* 曝光特征
* 历史特征
* 完播率
* 过去7天历史转化率
3. 生成user的embedding和feed的embedding
4. 获取user的历史行为序列

数据集：
1. features/：存储的是获取统计特征后的用户和视频特征，以及用户历史行为序列
2. embeddings/：使用deepwalk生成的feed embeddings以及基于用户观看的视频生成的user embeddings
"""

import pandas as pd
import numpy as np
import os, copy, importlib, feature_utils, feature_embedding, json
from feature_utils import *
from feature_embedding import *
importlib.reload(feature_utils)
importlib.reload(feature_embedding)
from tensorflow.keras.preprocessing.sequence import pad_sequences
pd.set_option('display.max_columns', None)


target = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow"]
more_target = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow", "stay", "play"]
tag_features = ['manual_keyword_list_1', 'machine_keyword_list_1', 'manual_tag_list_1', 'machine_tag_list_1'
        , 'manual_keyword_list_2', 'machine_keyword_list_2', 'manual_tag_list_2', 'machine_tag_list_2'
        , 'manual_keyword_list_3', 'machine_keyword_list_3', 'manual_tag_list_3', 'machine_tag_list_3']
sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id'] + tag_features
dense_features = ['videoplayseconds']
merge_features = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id'] + tag_features


def preprocess_data():
    """处理feed和user数据并生成统计特征"""
    feed = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/feed_info.csv')
    action = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/user_action.csv')
    test = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/test_a.csv')

    feed = preprocess_feed(feed)
    feed = preprocess_videoplayseconds(feed)
    data = pd.merge(action, feed, on='feedid')
    # print("data的信息:", data.head())
    data, user_features, video_features = generate_statistical_features(data)
    # data = pd.read_csv('./data/data.csv')
    # video_features = pd.read_csv('./data/feed_features.csv')
    # user_features = pd.read_csv('./data/user_features.csv')
    print("data:\n")
    print(data.info())
    print("user_features:\n")
    print(user_features.info())
    print("video_features:\n")
    print(video_features.info())
    
    return data, user_features, video_features

def get_feed_embedding():
    """获取视频的embedding"""
    data, _, video_features = preprocess_data()
    data = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/data.csv')
    data = data.dropna(subset=['userid', 'feedid']) # 用户交互历史

    print("================== 开始处理feed的embeddings ================")
    deepwalk_feed_embedding = generate_deepwalk_embedding(data, video_features)
    # 这里不能取消掉读取数据的注释，否则会出现报错
    # deepwalk_feed_embedding = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/embeddings/deepwalk_feed_embedding.csv')
    # print("deepwalk_feed_embedding:", deepwalk_feed_embedding.head(5))
    # def str_to_array(emb_str):
    #     return np.array([float(num) for num in emb_str.strip('[]').split()])
    # deepwalk_feed_embedding['deepwalk_embedding']=deepwalk_feed_embedding['deepwalk_embedding'].apply(str_to_array)
    # deepwalk_feed_embedding.to_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/embeddings/deepwalk_embedding_transform.csv', index=False)

    # 处理feed_embeddings的格式
    feed_embeddings = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/feed_embeddings.csv')
    feed_embeddings['feed_embedding'] = feed_embeddings['feed_embedding'].apply(
        lambda x: [s for s in x.split(' ') if s.strip() != '']
    )
    feed_embeddings['feed_embedding'] = feed_embeddings['feed_embedding'].apply(
        lambda x: np.array([float(num) for num in x])
    )
    feed_embeddings.to_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/embeddings/deepwalk_feed_embedding.csv', index=False)  
    return deepwalk_feed_embedding, feed_embeddings # 返回deepwalk和多模态embedding


def get_user_embedding():
    """获取用户的embedding"""
    data = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/data.csv')
    data = data.dropna(subset=['userid', 'feedid']) # 用户交互历史
    deepwalk_feed_embedding = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/embeddings/deepwalk_feed_embedding.csv')
    print("================ 开始生成user的embedding =======================")
    # 根据用户观看的视频来生成user的embedding
    # 处理merged_data的feed embedding，否则为字符串的形式 
    merged_data = pd.merge(data, deepwalk_feed_embedding, on='feedid') # 这里不要加上how='left' 会导致值为空
    def str_to_array(emb_str):
        return np.array([float(num) for num in emb_str.strip('[]').split()])
    merged_data['feed_embedding']=merged_data['feed_embedding'].apply(str_to_array)
    # 此时merged_data的feed_embedding为字符串，需要转化为数组
    def average_embeddings(embeddings):
        valid_embeddings = [emb for emb in embeddings if isinstance(emb, np.ndarray)]
        return np.mean(valid_embeddings, axis=0)
    # 按 userid 分组计算平均 embedding
    user_embeddings = merged_data.groupby('userid')['feed_embedding'].apply(average_embeddings).reset_index()
    user_embeddings.rename(columns={'feed_embedding': 'user_embedding'}, inplace=True) # 修改列名
    user_embeddings.to_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/embeddings/user_embedding.csv', index=False)
    print("通过deepwalk_embedding生成的user embedding已经存储")
    
    print("============= 开始生成用户的历史行为序列 ========================")
    user_history = build_user_history_sequences(data)
    # 处理可能存在的 NumPy 数组
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    user_history_serializable = {
        user_id: {
            key: {
                sub_key: convert_to_serializable(sub_val)
                if isinstance(sub_val, np.ndarray) else sub_val
                for sub_key, sub_val in val.items()
            } if isinstance(val, dict) else convert_to_serializable(val)
            for key, val in history.items()
        }
        for user_id, history in user_history.items()
    }
    # 存储为 JSON 文件
    with open('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/features/user_history_sequences.json', 'w', encoding='utf-8') as f:
        json.dump(user_history_serializable, f, ensure_ascii=False, indent=4)
    print("用户历史行为序列已存储")
    
    return user_embeddings, user_history # 返回用户的embedding和用户历史行为序列


def user_history_to_dataframe(user_history):
    records = []
    
    for userid, history in user_history.items():
        record = {'userid': float(userid)} # 需要确保为float，避免后续拼接出现问题
        
        # 各目标行为队列
        for target, feed_list in history['target_behavior_queues'].items():
            col_name = f"{target}_target_behavior_feed"
            record[col_name] = str(feed_list)  # 转为字符串形式，CSV 中仍可还原为 list
        
        # 其他行为队列
        record['interactive_history'] = str(history['interactive_history'])
        record['non_interactive_history'] = str(history['non_interactive_history'])
        record['finish_history'] = str(history['finish_history'])
        record['unfinish_history'] = str(history['unfinish_history'])
        
        # 每日展示历史
        record['daily_show_history'] = str(history['daily_show_history'])  # 如果不想保留可删除
        
        records.append(record)
    
    return pd.DataFrame(records)


# # 获取feed相关的embedding
# deepwalk_feed_embedding, feed_embeddings = get_feed_embedding()
# # 获取user相关的embedding
# user_embeddings, user_history = get_user_embedding()

# preprocess_data()