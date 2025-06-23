# -*- coding: utf-8 -*-

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


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import os, copy, importlib, feature_embedding, json, joblib, feature, math, pickle, utils, gc
from feature_embedding import *
importlib.reload(feature_embedding)
importlib.reload(feature)
importlib.reload(utils)
from feature import *
from utils import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
pd.set_option('display.max_columns', None)
from tqdm import tqdm

target = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow"]
more_target = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow", "stay", "play"]
tag_features = ['manual_keyword_list_1', 'machine_keyword_list_1', 'manual_tag_list_1', 'machine_tag_list_1'
        , 'manual_keyword_list_2', 'machine_keyword_list_2', 'manual_tag_list_2', 'machine_tag_list_2'
        , 'manual_keyword_list_3', 'machine_keyword_list_3', 'manual_tag_list_3', 'machine_tag_list_3']
sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id'] + tag_features
dense_features = ['videoplayseconds']
merge_features = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id'] + tag_features



def preprocess_feed(feed, tf_config):
    """处理feed数据，包括填充数据，以及对多值id进行离散化、tfidf以及svd降维"""
    # ========处理feed数据=======
    # 1. 填充缺失值
    feed[["bgm_song_id", "bgm_singer_id", "authorid"]] += 1 # 0用于表示未知
    feed[["bgm_song_id", "bgm_singer_id", "authorid", "videoplayseconds"]] = \
        feed[["bgm_song_id", "bgm_singer_id", "authorid", "videoplayseconds"]].fillna(0)
    feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
    feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')
    feed['manual_tag_list'] = feed['manual_tag_list'].fillna('-1') 
    feed['machine_tag_list'] = feed['machine_tag_list'].fillna('-1')
    feed['manual_keyword_list'] = feed['manual_keyword_list'].fillna('-1')
    feed['machine_keyword_list'] = feed['machine_keyword_list'].fillna('-1')
    for name in ['description', 'ocr', 'asr', 'description_char', 'ocr_char', 'asr_char']:
        feed[name] = feed[name].fillna('-1')
    
    # 2. 处理多值id：manual_tag_list, machine_tag_list, manual_keyword_list, machine_keyword_list
    # （1）按照其分割符拆分，拆分为数组
    def split(column):
        if not isinstance(column,str):
            return []
        keys = column.strip().split(';')
        for key in keys:
            if key not in key2index:
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], keys))
    reverse_manual = {}  # 人工标签/关键词反向映射
    reverse_machine = {}  # 机器标签/关键词反向映射
    # （1.1）处理manual_keyword_list, manual_tag_list
    global key2index_manual
    encoder_manual = {}
    for f in ['manual_keyword_list', 'manual_tag_list']:  # 修正字段名
        key2index_manual = {}
        f_list = []
        
        # 显式遍历每一行，处理可能的异常格式
        for idx, column in enumerate(feed[f].values):
            if not isinstance(column, str) or pd.isna(column):
                f_list.append([])
                continue
                
            keys = column.strip().split(';')
            for key in keys:
                if key not in key2index_manual:
                    key2index_manual[key] = len(key2index_manual) + 1
            f_list.append(list(map(lambda x: key2index_manual[x], keys)))
        
        f_length = np.array(list(map(len, f_list)))
        max_len = max(f_length)
        # print(f'{f} 字段最长的取值序列长度为 {max_len}')
        
        # 确保pad_sequences输入是列表的列表
        padded_sequences = pad_sequences(f_list, maxlen=max_len, padding='post')
        
        # 保存填充后的序列
        feed[f] = list(padded_sequences)
        encoder_manual[f] = copy.copy(key2index_manual)

        # 人工字段反向映射（数值→文本）
        reverse_manual[f] = {v: k for k, v in key2index_manual.items()}

    # （1.2）处理machine_keyword_list, machine_tag_list
    global key2index_machine
    encoder_machine = {}
    for f in ['machine_keyword_list', 'machine_tag_list']:  # 修正字段名
        key2index_machine = {}
        f_list = []
        
        # 显式遍历每一行，处理可能的异常格式
        for idx, column in enumerate(feed[f].values):
            if not isinstance(column, str) or pd.isna(column):
                f_list.append([])
                continue
                
            keys = column.strip().split(';')
            for key in keys:
                if key not in key2index_machine:
                    key2index_machine[key] = len(key2index_machine) + 1
            f_list.append(list(map(lambda x: key2index_machine[x], keys)))
        
        f_length = np.array(list(map(len, f_list)))
        max_len = max(f_length)
        # print(f'{f} 字段最长的取值序列长度为 {max_len}')
        
        # 确保pad_sequences输入是列表的列表
        padded_sequences = pad_sequences(f_list, maxlen=max_len, padding='post')
        
        # 保存填充后的序列
        feed[f] = list(padded_sequences)
        encoder_machine[f] = copy.copy(key2index_machine)

        # 机器字段反向映射（数值→文本）
        reverse_machine[f] = {v: k for k, v in key2index_machine.items()}

    # （1.3）创建文件夹并保存编码字典
    os.makedirs(tf_config['encoder_path'], exist_ok=True)
    joblib.dump(encoder_machine, tf_config['encoder_machine_path'])
    joblib.dump(encoder_manual, tf_config['encoder_manual_path'])
    
    # （2）使用tf-idf处理
     # ========计算TF-IDF特征========
    tfidf_fields = [
        ('manual_tag_list', reverse_manual),
        ('manual_keyword_list', reverse_manual),
        ('machine_tag_list', reverse_machine),
        ('machine_keyword_list', reverse_machine)
    ]
    
    for field, reverse_mapping in tfidf_fields:
        if field not in reverse_mapping:
            print("警告：未找到{}的反向映射，跳过TF-IDF计算".format(field))
            continue
        
        key2text = reverse_mapping[field]
        text_corpus = []
        
        for seq in feed[field].values:
            # 过滤填充值（0）并转换为文本
            text = ' '.join([key2text.get(idx, '') for idx in seq if idx != 0])
            text_corpus.append(text)
        
        # 计算TF-IDF（这里不需要合并到原数据，否则维度过高导致模型训练变慢）
        vectorizer = TfidfVectorizer(min_df=2, max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(text_corpus)

        # （3）压缩tf-idf向量为低维稠密向量
        svd_components = 5  # 降维目标维度，可调整（这里先暂时设置为5）
        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        svd_matrix = svd.fit_transform(tfidf_matrix)
        
        # 计算解释方差比例（评估降维效果）
        explained_variance = svd.explained_variance_ratio_.sum()
        # print(f"{field} SVD降维后解释方差比例: {explained_variance:.4f}")
        
        # 将SVD结果转换为DataFrame
        svd_df = pd.DataFrame(
            svd_matrix,
            columns=["svd_{}_{}".format(field, i+1) for i in range(svd_components)]
        )
        feed = pd.concat([feed, svd_df], axis=1)
        
        # 保存SVD模型
        with open(os.path.join(tf_config['encoder_path'],f'svd_{field}.pkl'), 'wb') as f:   
            pickle.dump(svd, f)
        
        # 保存TF-IDF向量器
        with open(os.path.join(tf_config['encoder_path'],f'tfidf_{field}.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
    return feed


def preprocess_videoplayseconds(feed):
    """
    对视频长度进行离散化，分桶处理
    做离散化是因为播放时长有长尾分布，分桶后每个区间的样本量均匀，避免模型被超长视频主导
    """
    print("正在对videoplayseconds进行分桶离散化...")
    field = 'videoplayseconds'
    # # 1. 查看数据分布
    # print(f"\n{field} 数据分布统计：")
    # print(feed[field].describe())
    
    # 2. 定义分桶策略
    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 0%, 20%, 40%, 60%, 80%, 100%分位数
    bucket_labels = ['极短', '短', '中等', '长', '极长']  # 桶标签
    
    # 3. 计算分位数边界
    boundaries = feed[field].quantile(quantiles).tolist()
    # print(f"{field} 分桶边界：{boundaries}")
    
    # 4. 执行分桶
    feed["{}_bin".format(field)] = pd.cut(
        feed[field],
        bins=boundaries,
        labels=bucket_labels,
        include_lowest=True
    )
    
    # 5. 查看分桶结果分布
    bucket_counts = feed["{}_bin".format(field)].value_counts()
    # print(f"\n{field} 分桶结果分布：")
    # print(bucket_counts)
    
    # 6. 转为数值编码（便于模型处理）
    oe = OrdinalEncoder()
    feed["{}_bin_encoded".format(field)] = oe.fit_transform(feed[["{}_bin".format(field)]])
    
    # 7. 保存编码器
    os.makedirs('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/encoders', exist_ok=True)
    with open('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/encoders/ordinal_{}.pkl'.format(field), 'wb') as f:
        pickle.dump(oe, f)
    
    return feed 


def generate_statistical_features(data):
    """生成用户和视频维度的统计特征（基于数字日期）"""
    print("开始生成统计特征")
    # 获取数据中的最大日期作为参考点
    max_date = data['date_'].max()
    
    # ======== Nunique特征 ========
    print("正在生成unique特征...")
    # 用户维度：过去7天内（date >= max_date-7）浏览的不同feedid数量
    user_interest_breadth = data[data['date_'] >= max_date - 7].groupby('userid')['feedid'].nunique().reset_index()
    user_interest_breadth.rename(columns={'feedid': 'user_7d_feed_nunique'}, inplace=True)
    
    # 视频维度：被不同userid播放的数量
    video_coverage = data.groupby('feedid')['userid'].nunique().reset_index()
    video_coverage.rename(columns={'userid': 'video_user_nunique'}, inplace=True)
    
    # ======== exposure特征 ========
    print("正在生成exposure特征...")
    # 用户维度：过去14天内（date >= max_date-14）的曝光次数
    user_exposure = data[data['date_'] >= max_date - 14].groupby('userid').size().reset_index(name='user_14d_exposure')
    
    # 视频维度：过去14天内（date >= max_date-14）的曝光次数
    video_exposure = data[data['date_'] >= max_date - 14].groupby('feedid').size().reset_index(name='video_14d_exposure')
    
    # ======== history特征 ========
    print("正在生成history特征...")
    # 行为列列表
    action_columns = ['read_comment', 'comment', 'like', 'play', 'stay', 
                     'click_avatar', 'forward', 'follow', 'favorite']
    
    # 用户维度：过去7天内（date >= max_date-7）每个行为的统计量
    user_history = data[data['date_'] >= max_date - 7].groupby('userid')[action_columns].agg(['sum', 'var', 'mean', 'median']).reset_index()
    if isinstance(user_history.columns, pd.MultiIndex):
        new_columns = []
        for col in user_history.columns:
            if col[0] == 'userid' and col[1] == '':
                new_columns.append('userid')
            else:
                new_columns.append("user_{}_{}".format(col[0], col[1]))
        user_history.columns = new_columns
        
    # 视频维度：过去7天内（date >= max_date-7）每个行为的统计量
    video_history = data[data['date_'] >= max_date - 7].groupby('feedid')[action_columns].agg(['sum', 'var', 'mean', 'median']).reset_index()
    if isinstance(video_history.columns, pd.MultiIndex):
        new_columns = []
        for col in video_history.columns:
            if col[0] == 'feedid' and col[1] == '':
                new_columns.append('feedid')
            else:
                new_columns.append("feed_{}_{}".format(col[0], col[1]))
        video_history.columns = new_columns    
    
    # ======== 完播率特征 ========
    print("正在生成完播率特征...")
    # 计算每个播放记录是否完播（播放时长 >= 视频时长）
    data['is_complete'] = data['stay'] >= data['videoplayseconds']
    
    # 视频维度：完播率
    video_complete_rate = data.groupby('feedid')['is_complete'].mean().reset_index(name='video_complete_rate')
    
    # 用户维度：平均完播率
    user_complete_rate = data.groupby('userid')['is_complete'].mean().reset_index(name='user_complete_rate')
    
    # ======== 历史转化率特征 ========
    print("正在生成历史转化率特征...")
    # 用户维度：过去7天内（date >= max_date-7）like=1的播放次数/总播放次数
    user_like_data = data[data['date_'] >= max_date - 7]
    user_like_count = user_like_data.groupby('userid')['like'].sum().reset_index()
    user_play_count = user_like_data.groupby('userid')['play'].sum().reset_index()
    user_conversion = pd.merge(user_like_count, user_play_count, on='userid', how='left')
    user_conversion['user_like_conversion'] = user_conversion['like'] / user_conversion['play'].replace(0, np.nan)
    user_conversion = user_conversion[['userid', 'user_like_conversion']]
    
    # 视频维度：过去7天内（date >= max_date-7）like=1的播放次数/总播放次数
    video_like_count = user_like_data.groupby('feedid')['like'].sum().reset_index()
    video_play_count = user_like_data.groupby('feedid')['play'].sum().reset_index()
    video_conversion = pd.merge(video_like_count, video_play_count, on='feedid', how='left')
    video_conversion['video_like_conversion'] = video_conversion['like'] / video_conversion['play'].replace(0, np.nan)
    video_conversion = video_conversion[['feedid', 'video_like_conversion']]
    
    # ======== 合并所有特征 ========
    print("正在合并所有特征...")
    # 合并用户维度特征
    user_features = user_interest_breadth
    for df in [user_exposure, user_history, user_complete_rate, user_conversion]:
        user_features = pd.merge(user_features, df, on='userid', how='right')
    
    # 合并视频维度特征
    video_features = video_coverage
    # print("video_features:", video_features)
    for df in [video_exposure, video_history, video_complete_rate, video_conversion]:
        # print(df.columns)
        video_features = pd.merge(video_features, df, on='feedid', how='right')

    return data, user_features, video_features


def build_user_history_sequences(data):
    """用户的历史行为序列建模"""
    targets = ["read_comment", "like", "click_avatar", "forward", "favorite", "comment", "follow"]

    user_history = {}
    # 按用户 ID 分组
    grouped = data.groupby('userid')
    
    for userid, group in tqdm(grouped, desc='构建用户历史行为序列进度'):
        # 按 date_ 对用户记录进行排序
        group = group.sort_values(by='date_')
        
        # 初始化 5 种历史队列
        target_behavior_queues = {target: [] for target in targets}
        interactive_history = []
        non_interactive_history = []
        finish_history = []
        unfinish_history = []
        daily_show_history = {}  # 改为字典，按日期存储feed
        
        # 遍历用户的每条记录
        for index, row in group.iterrows():
            date = row['date_']
            # 7 个目标行为分别对应的历史队列
            for target in targets:
                if row[target] == 1:
                    target_behavior_queues[target].append(row['feedid'])
            
            # 有交互行为的历史队列
            if any(row[target] == 1 for target in targets):
                interactive_history.append(row['feedid'])
            else:
                # 展现但是没有触发交互行为的历史队列
                non_interactive_history.append(row['feedid'])
            
            # Finish 和 UnFinish 的历史队列
            is_complete = row['stay'] >= row['videoplayseconds']
            if is_complete:
                finish_history.append(row['feedid'])
            else:
                unfinish_history.append(row['feedid'])
            
            # 用户当天展现视频队列
            if date not in daily_show_history:
                daily_show_history[date] = []
            daily_show_history[date].append(row['feedid'])

        user_history[userid] = {
            'target_behavior_queues': target_behavior_queues,
            'interactive_history': interactive_history,
            'non_interactive_history': non_interactive_history,
            'finish_history': finish_history,
            'unfinish_history': unfinish_history,
            'daily_show_history': daily_show_history
        }

    return user_history

def preprocess_data(feed, action, tf_config):
    """处理feed和user数据并生成统计特征"""
    print("########## 处理数据前 ############")
    # action中并不是所有的电影都出现了feed中，feed中并不是所有电影都出现在action中
    # 填充缺失值
    # feed = feed.fillna({
    #     'bgm_song_id': 0,
    #     'bgm_singer_id': 0,
    #     'videoplayseconds': 0,
    #     'manual_tag_list': '-1',
    #     'machine_tag_list': '-1',
    #     'manual_keyword_list': '-1',
    #     'machine_keyword_list': '-1',
    #     'description': '-1',
    #     'ocr': '-1',
    #     'asr': '-1'
    # })
    # test = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/test_a.csv')
    print("处理前的user-feed交互数量为{}，其中参与交互的用户数量为{}，参与交互的视频号数量为{}".format(len(action), action['userid'].nunique(), action['feedid'].nunique()))
    print("处理前的feed_info中的feed数量为{}".format(feed['feedid'].nunique()))
    
    #### 检查数据情况
    # 获取action中所有唯一的feedid
    action_feedids = set(action['feedid'].unique())
    feed_feedids = set(feed['feedid'].unique())
    # 找出在action中出现但不在feed中的feedid
    missing_feedids = feed_feedids - action_feedids
    missing_feedids_1= action_feedids - feed_feedids
    if len(missing_feedids)>0:
        print("警告: 有{}个feedid在feed中出现但不在action中".format(len(missing_feedids)))
    if len(missing_feedids_1)>0:
        print("警告: 有{}个feedid在action中出现但不在feed中".format(len(missing_feedids_1)))
    del missing_feedids
    del missing_feedids_1
    print("########## 处理数据后 ###############")
    feed = preprocess_feed(feed, tf_config) 
    feed = preprocess_videoplayseconds(feed)
    data = pd.merge(action, feed, on='feedid')
    data, user_features, video_features = generate_statistical_features(data)
    # 加上authorid 否则后续读取不到

    data = data.dropna(subset=['userid', 'feedid'])
    print("处理数据后，user-feed交互数量为{}，其中参与交互的用户数量为{}，参与交互的视频号数量为{}".format(len(data), data['userid'].nunique(), data['feedid'].nunique()))
    print("同时，收集到features的user数量为{}，feed数量为{}".format(user_features['userid'].nunique(), video_features['feedid'].nunique()))
    
    print("data:\n")
    print(data.info())
    print("user_features:\n")
    print(user_features.info())
    print("video_features:\n")
    print(video_features.info())
    
    return data, user_features, video_features

def get_feed_embedding(data, video_features, feed_embeddings):
    """获取视频的embedding"""
    print("================== 开始生成feed的embeddings ================")
    word2vec_feed_embedding = generate_word2vec_embedding(data, video_features)
    return word2vec_feed_embedding, feed_embeddings # 返回deepwalk和多模态embedding


# def get_feed_embedding(data, video_features, feed_embeddings):
#     """获取视频的embedding"""
#     print("================== 开始生成feed的embeddings ================")
#     word2vec_feed_embedding = generate_word2vec_embedding(data, video_features)

#     # 将 Word2Vec 嵌入和多模态特征转换为 numpy 数组
#     word2vec_emb_list = np.array([np.array(emb) for emb in word2vec_feed_embedding['feed_word2vec_embedding']])
#     multi_modal_emb_list = np.array(feed_embeddings)

#     # 扩展维度以满足多头注意力输入要求 [batch_size, seq_length, embedding_dim]
#     word2vec_emb_list = np.expand_dims(word2vec_emb_list, axis=1)
#     multi_modal_emb_list = np.expand_dims(multi_modal_emb_list, axis=1)

#     # 定义多头注意力层
#     num_heads = 4
#     embed_dim = max(word2vec_emb_list.shape[-1], multi_modal_emb_list.shape[-1])
#     attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

#     # 执行注意力操作
#     attn_output, _ = attention_layer(query=word2vec_emb_list, key=multi_modal_emb_list, value=multi_modal_emb_list)

#     # 合并注意力输出和原始 Word2Vec 嵌入
#     attn_output = np.squeeze(attn_output, axis=1)
#     word2vec_emb_list = np.squeeze(word2vec_emb_list, axis=1)
#     combined_embeddings = np.concatenate([word2vec_emb_list, attn_output], axis=1)

#     combined_embeddings = pd.DataFrame(combined_embeddings)
#     return combined_embeddings


def get_user_embedding(data, word2vec_feed_embedding, tf_config):
    """获取用户的embedding"""
    print("================ 开始生成user的embedding =======================")
    # 根据用户观看的视频来生成user的embedding
    # 如果是从csv中读取的数据，那么需要处理merged_data的feed embedding，否则为字符串的形式 
    merged_data = pd.merge(data, word2vec_feed_embedding, on='feedid') # 这里不要加上how='left'或者how='right' 会导致值为空，从而导致后续获取user的embedding失败
    merged_data['feed_word2vec_embedding']=merged_data['feed_word2vec_embedding'].apply(str_to_array)
    def average_embeddings(embeddings):
        valid_embeddings = [emb for emb in embeddings if isinstance(emb, np.ndarray)]
        return np.mean(valid_embeddings, axis=0)
    # 按 userid 分组计算平均 embedding
    user_embeddings = merged_data.groupby('userid')['feed_word2vec_embedding'].apply(average_embeddings).reset_index()
    user_embeddings.rename(columns={'feed_word2vec_embedding': 'user_embedding'}, inplace=True) # 修改列名
    print("通过word2vec_embedding生成的user embedding已成功")
    
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
    os.makedirs(os.path.dirname(tf_config['features_path']), exist_ok=True)
    with open(tf_config['user_history_sequences_path'], 'w', encoding='utf-8') as f:
        json.dump(user_history_serializable, f, ensure_ascii=False, indent=4)
    print("用户历史行为序列已存储")
    del user_history_serializable
    return user_embeddings, user_history # 返回用户的embedding和用户历史行为序列



def get_author_embedding(data):
    print("========== 开始生成author的embedding ===========")
    author_embedding = generate_author_embedding(data)
    print("author的embedding生成成功！")
    return author_embedding

def user_history_to_dataframe(user_history):
    records = []
    
    for userid, history in user_history.items():
        record = {'userid': float(userid)} # 需要确保为float，避免后续拼接出现问题
        
        # 各目标行为队列
        for target, feed_list in history['target_behavior_queues'].items():
            col_name = "{}_target_behavior_feed".format(target)
            record[col_name] = str(feed_list)  # 转为字符串形式，CSV 中仍可还原为 list
        
        # 其他行为队列
        record['interactive_history'] = str(history['interactive_history'])
        record['non_interactive_history'] = str(history['non_interactive_history'])
        record['finish_history'] = str(history['finish_history'])
        record['unfinish_history'] = str(history['unfinish_history'])
        
        # 每日展示历史
        record['daily_show_history'] = str(history['daily_show_history'])  # 如果不想保留可删除
        records.append(record)
        del record # 释放内存
    print("用户的历史交互序列处理成功")
    return pd.DataFrame(records)

def model_input(data, batch_size=200000):
    """处理数据为模型的输入格式（优化版，避免OOM）"""
    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    
    print(f"总共有 {num_batches} 个 batch")
    processed_batches = []

    for i in range(num_batches):
        print(f"\n正在处理第 {i+1} 个 batch")
        start = i * batch_size
        end = start + batch_size
        batch_data = data[start:end].copy()  # 避免 SettingWithCopyWarning

        # ======== 数值型特征 =========
        print("Step1: 处理数值型特征")
        batch_data[numeric_features] = batch_data[numeric_features].astype(np.float32)

        # ======== 类别型特征 =========
        print("Step2: 处理类别型特征")
        batch_data['is_complete'] = batch_data['is_complete'].astype(int)
        batch_data[categorical_features] = batch_data[categorical_features].astype(np.float32)

        # ======== 序列特征 =========
        print("Step3: 处理序列型特征")

        def split_and_pad(series):
            series = series.fillna('').astype(str).str.split().apply(lambda x: [int(i) for i in x] if x else [])
            max_len = series.apply(len).max()
            return series.apply(lambda x: x + [0] * (max_len - len(x)))

        # 描述文本
        print(" - description")
        batch_data['description'] = split_and_pad(batch_data['description'])

        # OCR / ASR 特征
        for col in ['ocr', 'asr']:
            print(f" - {col}")
            batch_data[col] = batch_data[col].fillna('').replace('-1', '').astype(str).str.split().apply(
                lambda x: [int(i) for i in x if i.strip().isdigit()] if x else []
            )
            max_len = batch_data[col].apply(len).max()
            batch_data[col] = batch_data[col].apply(lambda x: x + [0] * (max_len - len(x)))

        # 关键词与标签类特征
        for col in ["manual_keyword_list", "machine_keyword_list", "manual_tag_list", "machine_tag_list"]:
            print(f" - {col}")
            batch_data[col] = batch_data[col].fillna('').astype(str).str.strip('[]').str.split().apply(
                lambda x: [int(i) for i in x if i.strip().isdigit()] if x else []
            )
            max_len = batch_data[col].apply(len).max()
            batch_data[col] = batch_data[col].apply(lambda x: x + [0] * (max_len - len(x)))

        # 行为序列类特征
        for col in [
            "read_comment_target_behavior_feed",
            "like_target_behavior_feed",
            "click_avatar_target_behavior_feed",
            "forward_target_behavior_feed",
            "follow_target_behavior_feed",
            "favorite_target_behavior_feed",
            "comment_target_behavior_feed",
            "interactive_history",
            "non_interactive_history",
            "finish_history",
            "unfinish_history"
        ]:
            print(f" - {col}")
            batch_data[col] = batch_data[col].fillna('').astype(str).str.strip('[]').str.split(',').apply(
                lambda x: [int(i.strip()) for i in x if i.strip().isdigit()] if x else []
            )
            max_len = batch_data[col].apply(len).max()
            batch_data[col] = batch_data[col].apply(lambda x: x + [0] * (max_len - len(x)))

        # 加入当前 batch 到总结果中
        processed_batches.append(batch_data)

        # 清理内存
        del batch_data
        gc.collect()

    print("\n拼接所有 batch...")
    final_data = pd.concat(processed_batches, ignore_index=True)
    del processed_batches
    gc.collect()

    print("model_input 处理完成！")
    return final_data


# def model_input(data, batch_size=200000):
#     """处理数据为模型的输入格式"""
#     num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    
#     # 假设这些变量在函数外部已经定义
#     global numeric_features, categorical_features
#     print(f"总共有{num_batches}个batch")
#     for i in range(num_batches):
#         print(f"正在处理第 {i+1} 个 batch")
#         start = i * batch_size
#         end = start + batch_size
#         batch_data = data[start:end].copy()  # 避免 SettingWithCopyWarning
        
#         print("======== 正在处理数值型特征 ===========")
#         # 将numeric_features变成float32
#         batch_data[numeric_features] = batch_data[numeric_features].astype(np.float32)

#         print("======== 正在处理类别型特征 ===========")
#         batch_data['is_complete'] = batch_data['is_complete'].astype(int)
#         batch_data[categorical_features]=batch_data[categorical_features].astype(np.float32)

#         print("======== 开始处理序列型特征 ===========")
#         print("Step1: 处理description")
#         # 1. 处理description
#         def split_and_pad(series):
#             series = series.str.split().apply(lambda x: [int(i) for i in x] if x else [])
#             max_len = series.apply(len).max()
#             return series.apply(lambda x: x + [0] * (max_len - len(x)))
        
#         batch_data['description'] = split_and_pad(batch_data['description'])

#         print("Step2: 处理ocr和asr")
#         # 2. 处理ocr和asr
#         for col in ['ocr', 'asr']:
#             batch_data[col] = batch_data[col].replace('-1', '').str.split().apply(lambda x: [int(i) for i in x] if x else [])
#             max_len = batch_data[col].apply(len).max()
#             batch_data[col] = batch_data[col].apply(lambda x: [0] * max_len if not x else x + [0] * (max_len - len(x)))

#         print("Step3: 处理keyword和tag的manual和machine特征")
#         # 3. 处理manual_keyword_list, machine_keyword_list, manual_tag_list, machine_tag_list
#         for col in ["manual_keyword_list", "machine_keyword_list", "manual_tag_list", "machine_tag_list"]:
#             batch_data[col] = batch_data[col].astype(str).str.strip('[]').str.split().apply(lambda x: [int(i) for i in x] if x else [])
#             max_len = batch_data[col].apply(len).max()
#             batch_data[col] = batch_data[col].apply(lambda x: x + [0] * (max_len - len(x)))

#         print("Step4: 处理剩下的序列特征")
#         # 4. 处理剩下的序列特征
#         for col in ["read_comment_target_behavior_feed",
#                     "like_target_behavior_feed",
#                     "click_avatar_target_behavior_feed",
#                     "forward_target_behavior_feed",
#                     "follow_target_behavior_feed",
#                     "favorite_target_behavior_feed",
#                     "comment_target_behavior_feed",
#                     "interactive_history",
#                     "non_interactive_history",
#                     "finish_history",
#                     "unfinish_history"]:
#             batch_data[col] = batch_data[col].astype(str).str.strip('[]').str.split(',').apply(lambda x: [int(i.strip()) for i in x if i.strip()] if x else [])
#             max_len = batch_data[col].apply(len).max()
#             batch_data[col] = batch_data[col].apply(lambda x: [0] * max_len if not x else x + [0] * (max_len - len(x)))
        
#         data[start:end] = batch_data
#         # 手动释放 batch_data 的内存
#         del batch_data
    
#     return data

# def model_input(data):
#     """处理数据为模型的输入格式"""
#     print("======== 正在处理数值型特征 ===========")
#     # 将numeric_features变成float32
#     data[numeric_features] = data[numeric_features].astype(float)

#     # 将类别型变成float32
#     print("======== 正在处理类别型特征 ===========")
#     data['is_complete'] = data['is_complete'].astype(int)
#     data[categorical_features]=data[categorical_features].astype(float)

#     print("======== 开始处理序列型特征 ===========")
#     print("Step1: 处理description")
#     # 1. 处理description
#     data['description'] = data['description'].apply(lambda x: list(map(int, x.split())))
#     max_len_description = data['description'].apply(len).max()
#     data['description'] = data['description'].apply(
#         lambda x: x + [0] * (max_len_description - len(x)) if len(x) < max_len_description else x
#     )

#     # 2. 处理ocr和asr
#     print("Step2: 处理ocr和asr")
#     for col in ['ocr', 'asr']:
#         data[col] = data[col].apply(
#             lambda x: [] if x == '-1' else list(map(int, x.split()))
#         )
#         max_len = data[col].apply(len).max()
#         data[col] = data[col].apply(
#             lambda x: [0] * max_len if len(x) == 0 else x + [0] * (max_len - len(x))
#         )

#     # 3. 处理manual_keyword_list, machine_keyword_list, manual_tag_list, machine_tag_list
#     print("Step3: 处理keyword和tag的manual和machine特征")
#     for col in ["manual_keyword_list","machine_keyword_list","manual_tag_list","machine_tag_list"]:
#         data[col] = data[col].astype(str)
#         # 先去掉两侧的空格，然后再按照空格进行分隔
#         data[col] = data[col].str.strip('[]').str.split().apply(
#         lambda x: list(map(int, x))
#     )

#     # 4. 处理剩下的序列特征
#     print("Step4: 处理剩下的序列特征")
#     for col in ["read_comment_target_behavior_feed",
#         "like_target_behavior_feed",
#         "click_avatar_target_behavior_feed",
#         "forward_target_behavior_feed",
#         "follow_target_behavior_feed",
#         "favorite_target_behavior_feed",
#         "comment_target_behavior_feed",
#         "interactive_history",
#         "non_interactive_history",
#         "finish_history",
#         "unfinish_history"]:
#         data[col] = data[col].str.strip('[]')
#         data[col] = data[col].apply(
#             lambda x: [] if x == '' else list(map(int, x.split(',')))
#         )
#         max_len = data[col].apply(len).max()
#         data[col] = data[col].apply(
#             lambda x: [0] * max_len if len(x) == 0 else x + [0] * (max_len - len(x))
#         )

#     return data


def get_features_config(data, feed, user_features):
    features_config={
        'dense': dense,
        'sequence': {},
        'sparse': {}
    }
    # 基于features文件中的特征生成配置文件
    for feat in sequence:
        print("正在生成{}的特征配置".format(feat))
        if isinstance(data[feat].iloc[0], str):  # 检查是否是字符串格式，因为将数组保存为csv文件时会自动解析为字符串，导致无法正确读取数组
            data[feat] = data[feat].str.strip('[]').str.split(',').apply(
                lambda x: list(map(int, x)) if x != [''] else []
            )
        max_len=data[feat].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
        
        vocab_size = max_len + 10  # 加缓冲值
        embedding_dim = 128 # 这里手动设置为128

        features_config['sequence'][feat]={
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'max_len': max_len
        }
    print('sequence feature的特征配置生成成功')
    
    for feat in sparse:
        print("正在生成{}的特征配置".format(feat))
        if feat=='userid':
            vocab_size = max(user_features['userid'])+10
            embedding_dim = len(data['user_embedding'][0])
        elif feat =='authorid':
            vocab_size = max(feed['authorid'])+10 # 注意这里不能通过data中的来获取，否则会导致后续embedding找不到对应的id
            embedding_dim = len(data['author_embedding'][0])
        elif feat=='feedid':
            vocab_size = max(feed['feedid'])+10
            embedding_dim = len(data['feed_word2vec_embedding'][0])
        else:
            vocab_size=data[feat].nunique()+1
            # embedding_dim = min(64, max(8, int(4 * (1 + math.log(vocab_size)))))
            embedding_dim = 128 # 手动设置为128

        features_config['sparse'][feat]={
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
        }
    print('sparse feature的特征配置生成成功')
    return features_config