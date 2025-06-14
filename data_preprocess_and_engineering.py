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
import os, copy, importlib, feature_utils
from feature_utils import *
importlib.reload(feature_utils)
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
    data = pd.merge(action, feed, on='feedid', how='right')
    print("data的信息:", data.info())
    user_features, video_features = generate_statistical_features(data)
    print("user_features的信息:", user_features.info())
    print("video_features:", video_features.info())


preprocess_data()
