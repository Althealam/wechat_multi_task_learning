import pandas as pd
import numpy as np
import os, copy, pickle, joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import OrdinalEncoder

def preprocess_feed(feed):
    """处理feed数据，包括填充数据，以及对多值id进行离散化、tfidf以及svd降维"""
    # ========处理feed数据=======
    # 1. 填充缺失值
    feed[["bgm_song_id", "bgm_singer_id"]] += 1 # 0用于表示未知
    feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
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
        print(f'{f} 字段最长的取值序列长度为 {max_len}')
        
        # 确保pad_sequences输入是列表的列表
        padded_sequences = pad_sequences(f_list, maxlen=max_len, padding='post')
        
        # 保存填充后的序列
        feed[f'{f}'] = list(padded_sequences)
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
        print(f'{f} 字段最长的取值序列长度为 {max_len}')
        
        # 确保pad_sequences输入是列表的列表
        padded_sequences = pad_sequences(f_list, maxlen=max_len, padding='post')
        
        # 保存填充后的序列
        feed[f'{f}'] = list(padded_sequences)
        encoder_machine[f] = copy.copy(key2index_machine)

        # 机器字段反向映射（数值→文本）
        reverse_machine[f] = {v: k for k, v in key2index_machine.items()}

    # （1.3）创建文件夹并保存编码字典
    os.makedirs('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/encoders', exist_ok=True)
    joblib.dump(encoder_machine, './data/encoders/encoder_machine.txt')
    joblib.dump(encoder_manual, './data/encoders/encoder_manual.txt')
    
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
            print(f"警告：未找到{field}的反向映射，跳过TF-IDF计算")
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
        print(f"{field} SVD降维后解释方差比例: {explained_variance:.4f}")
        
        # 将SVD结果转换为DataFrame
        svd_df = pd.DataFrame(
            svd_matrix,
            columns=[f"svd_{field}_{i+1}" for i in range(svd_components)]
        )
        feed = pd.concat([feed, svd_df], axis=1)
        
        # 保存SVD模型
        os.makedirs('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/encoders', exist_ok=True)
        with open(f'/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/encoders/svd_{field}.pkl', 'wb') as f:
            pickle.dump(svd, f)
        
        # 保存TF-IDF向量器
        with open(f'/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/encoders/tfidf_{field}.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    return feed


def preprocess_videoplayseconds(feed):
    """
    对视频长度进行离散化，分桶处理
    做离散化是因为播放时长有长尾分布，分桶后每个区间的样本量均匀，避免模型被超长视频主导
    """
    field = 'videoplayseconds'
    # 1. 查看数据分布
    print(f"\n{field} 数据分布统计：")
    print(feed[field].describe())
    
    # 2. 定义分桶策略
    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 0%, 20%, 40%, 60%, 80%, 100%分位数
    bucket_labels = ['极短', '短', '中等', '长', '极长']  # 桶标签
    
    # 3. 计算分位数边界
    boundaries = feed[field].quantile(quantiles).tolist()
    print(f"{field} 分桶边界：{boundaries}")
    
    # 4. 执行分桶
    feed[f"{field}_bin"] = pd.cut(
        feed[field],
        bins=boundaries,
        labels=bucket_labels,
        include_lowest=True
    )
    
    # 5. 查看分桶结果分布
    bucket_counts = feed[f"{field}_bin"].value_counts()
    print(f"\n{field} 分桶结果分布：")
    print(bucket_counts)
    
    # 6. 转为数值编码（便于模型处理）
    oe = OrdinalEncoder()
    feed[f"{field}_bin_encoded"] = oe.fit_transform(feed[[f"{field}_bin"]])
    
    # 7. 保存编码器
    os.makedirs('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/encoders', exist_ok=True)
    with open(f'/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/encoders/ordinal_{field}.pkl', 'wb') as f:
        pickle.dump(oe, f)
    
    return feed 


def generate_statistical_features(data):
    """生成用户和视频维度的统计特征（基于数字日期）"""
    # 获取数据中的最大日期作为参考点
    max_date = data['date_'].max()
    
    # ======== Nunique特征 ========
    # 用户维度：过去7天内（date >= max_date-7）浏览的不同feedid数量
    user_interest_breadth = data[data['date_'] >= max_date - 7].groupby('userid')['feedid'].nunique().reset_index()
    user_interest_breadth.rename(columns={'feedid': 'user_7d_feed_nunique'}, inplace=True)
    
    # 视频维度：被不同userid播放的数量
    video_coverage = data.groupby('feedid')['userid'].nunique().reset_index()
    video_coverage.rename(columns={'userid': 'video_user_nunique'}, inplace=True)
    
    # ======== exposure特征 ========
    # 用户维度：过去14天内（date >= max_date-14）的曝光次数
    user_exposure = data[data['date_'] >= max_date - 14].groupby('userid').size().reset_index(name='user_14d_exposure')
    
    # 视频维度：过去14天内（date >= max_date-14）的曝光次数
    video_exposure = data[data['date_'] >= max_date - 14].groupby('feedid').size().reset_index(name='video_14d_exposure')
    
    # ======== history特征 ========
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
                new_columns.append(f"{col[0]}_{col[1]}")
        user_history.columns = new_columns
        
    # 视频维度：过去7天内（date >= max_date-7）每个行为的统计量
    video_history = data[data['date_'] >= max_date - 7].groupby('feedid')[action_columns].agg(['sum', 'var', 'mean', 'median']).reset_index()
    if isinstance(video_history.columns, pd.MultiIndex):
        new_columns = []
        for col in video_history.columns:
            if col[0] == 'feedid' and col[1] == '':
                new_columns.append('feedid')
            else:
                new_columns.append(f"{col[0]}_{col[1]}")
        video_history.columns = new_columns    
    # ======== 完播率特征 ========
    # 计算每个播放记录是否完播（播放时长 >= 视频时长）
    data['is_complete'] = data['stay'] >= data['videoplayseconds']
    
    # 视频维度：完播率
    video_complete_rate = data.groupby('feedid')['is_complete'].mean().reset_index(name='video_complete_rate')
    
    # 用户维度：平均完播率
    user_complete_rate = data.groupby('userid')['is_complete'].mean().reset_index(name='user_complete_rate')
    
    # ======== 历史转化率特征 ========
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
    # 合并用户维度特征
    user_features = user_interest_breadth
    for df in [user_exposure, user_history, user_complete_rate, user_conversion]:
        user_features = pd.merge(user_features, df, on='userid', how='right')
    
    # 合并视频维度特征
    video_features = video_coverage
    for df in [video_exposure, video_history, video_complete_rate, video_conversion]:
        video_features = pd.merge(video_features, df, on='feedid', how='right')
    
    return user_features, video_features
