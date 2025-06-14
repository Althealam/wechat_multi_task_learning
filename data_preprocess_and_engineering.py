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
"""

import pandas as pd
import numpy as np
import os, copy, importlib, feature_utils, feature_embedding
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
    feed = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/feed_info.csv')
    action = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/user_action.csv')
    test = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/test_a.csv')

    feed = preprocess_feed(feed)
    feed = preprocess_videoplayseconds(feed)
    # print("feed.head:", feed.head(5))
    # print("feed的列:", feed.columns)
    # print("action的列:", action.columns)
    data = pd.merge(action, feed, on='feedid', how='right')
    # data.to_csv('./data/data.csv') # 注意不要保存为user_action
    # print("data的信息:", data.head())
    data, user_features, video_features = generate_statistical_features(data)
    # print("user_features:", user_features.head(5))
    print("data:\n")
    print(data.info())
    print("user_features:\n")
    print(user_features.info())
    print("video_features:\n")
    print(video_features.info())
    
    return data, user_features, video_features

def get_embedding():
    data, user_features, video_features = preprocess_data()
    # data = pd.read_csv('./data/user_action.csv')
    print("data:\n", data.head(5))
    deepwalk_feed_embedding = generate_deepwalk_embedding(data, video_features)

get_embedding()