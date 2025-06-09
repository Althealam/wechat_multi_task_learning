import json
import pandas as pd
import numpy as np
from features import dense_features, sparse_features, varlen_features

def read_json_file(file_path):
    """读取JSON文件并返回其内容
    
    Args:
        file_path (str): JSON文件的路径
    
    Returns:
        dict or list: JSON文件的内容，如果出错则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式")
        print(f"详细错误: {e}")
        return None
    except PermissionError:
        print(f"错误: 没有权限读取文件 '{file_path}'")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None
    
def save_json_file(file_path, file):
    """存储json文件并将file存储到file_path路径"""
    with open(file_path, 'w') as f:
        json.dump(file, f, indent=4)
        print("文件保存成功!")


def generate_complete_embedding_config(df, encoder, encoder_description):
    """生成包含dense、sparse和sequence的完整特征配置"""
    
    # ===================== 1. 给定特征类型 =====================
    # 稀疏特征
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    # 变长序列特征
    varlen_features = ['manual_tag_list','manual_keyword_list', 'description'] 
    # 稠密特征
    dense_features = ['videoplayseconds']
    
    # ===================== 2. 构建特征配置字典 =====================
    config = {
        "dense": dense_features,
        "sparse": {},
        "sequence": {} 
    }
    
    # ===================== 3. 配置sparse_feature =====================
    for feat in sparse_features:
        # 计算唯一值数量
        vocab_size = df[feat].nunique()+1
        embed_dim = min(6, int(np.sqrt(vocab_size)))
        # 特殊处理高基数特征
        config["sparse"][feat] = {
                "vocab_size": vocab_size,  
                "embedding_dim": embed_dim
            }
    
    # ===================== 4. 配置varlen_features =====================
    for feat in varlen_features:
        if feat!='description':
            vocab_size = len(encoder[feat]) + 1  # 词汇表大小，表示特征空间中唯一元素的总数，用于确定embedding层的维度
            max_len = int(df[feat].apply(len).max()) # 变长序列的最大长度，也就是最长的列表中包含多少个元素
            embed_dim = min(6, int(np.sqrt(vocab_size))) # 将离散特征映射到连续向量空间后的向量维度
            config["sequence"][feat]={
                "vocab_size": vocab_size, 
                "max_len": max_len,
                "embedding_dim": embed_dim
            }
        else:
            vocab_size = len(encoder_description[feat])+1
            max_len = int(df[feat].apply(len).max())
            embed_dim = min(6, int(np.sqrt(vocab_size)))
            config["sequence"][feat]={
                "vocab_size": vocab_size, 
                "max_len": max_len,
                "embedding_dim": embed_dim
            }
    return config


def analyze_data(df, name="Dataset"):
    print(f"\n{'='*50}")
    print(f"分析数据集: {name}")
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print("\n前5行数据:")
    display(df.head())
    
    print("\n数据类型统计:")
    print(df.dtypes.value_counts())
    
    # 识别数值型和类别型特征
    numeric_features = []
    categorical_features = []
    
    for col in df.columns:
        # 数值型特征判断条件
        if pd.api.types.is_numeric_dtype(df[col]):
            # 排除高基数的数值型ID列
            if df[col].nunique() < 20 or col in ['videoplayseconds', 'ear', 'air']:
                numeric_features.append(col)
            else:
                # 高基数的数值列通常视为类别型
                categorical_features.append(col)
        else:
            # 非数值型视为类别型
            categorical_features.append(col)
    
    print("\n数值型特征(numeric_features):")
    print(numeric_features)
    
    print("\n类别型特征(categorical_features):")
    print(categorical_features)
    
    # 数值型特征的统计信息
    if numeric_features:
        print("\n数值型特征描述统计:")
        display(df[numeric_features].describe().T)
    
    # 类别型特征的统计信息
    if categorical_features:
        print("\n类别型特征唯一值统计:")
        for col in categorical_features:
            print(f"\n{col}:")
            print(f"唯一值数量: {df[col].nunique()}")
            print("前10个最常见值:")
            print(df[col].value_counts().head(10))
    
    # 缺失值统计
    print("\n缺失值统计:")
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    if len(missing_data) > 0:
        display(missing_data)
    else:
        print("没有缺失值")
    
    return numeric_features, categorical_features


def improved_analyze_data(df, name="Dataset"):
    print(f"\n{'='*50}")
    print(f"高级分析数据集: {name}")
    
    # 手动定义特征类型
    manual_numeric = ['videoplayseconds', 'play', 'stay', 'read_comment', 
                     'like', 'comment', 'click_avatar', 'forward', 'follow', 'favorite']
    manual_categorical = ['userid', 'feedid', 'device', 'authorid']
    manual_text = ['description', 'ocr', 'asr', 'manual_keyword_list', 
                  'machine_keyword_list', 'manual_tag_list', 'machine_tag_list']

    # 交叉验证自动检测和手动定义
    auto_numeric = []
    auto_categorical = []
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if col in manual_numeric or (df[col].nunique() > 10 and not col.endswith('_id')):
                auto_numeric.append(col)
            else:
                auto_categorical.append(col)
        else:
            auto_categorical.append(col)
    
    # 合并结果
    final_numeric = list(set(manual_numeric + auto_numeric))
    final_categorical = list(set(manual_categorical + auto_categorical))
    final_numeric = [x for x in final_numeric if x in df.columns]
    final_categorical = [x for x in final_categorical if x in df.columns]
    
    # 打印结果
    print("\n最终数值型特征:")
    print(final_numeric)
    
    print("\n最终类别型特征:")
    print(final_categorical)
    
    print("\n文本特征需要特殊处理:")
    print([x for x in manual_text if x in df.columns])
    
    print("\n高缺失值特征:")
    missing = df.isnull().sum()
    high_missing = missing[missing > len(df)*0.1].index.tolist()
    print(high_missing)
    
    return final_numeric, final_categorical

def preprocess(sample,dense_features):
    '''
    特征工程：对数值型特征做对数变换; id型特征+1; 缺失值补充0。
    '''
    sample[dense_features] = sample[dense_features].fillna(0.0)
    sample[dense_features] = np.log(sample[dense_features] + 1.0)
    
    sample[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = sample[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    sample["videoplayseconds"] = np.log(sample["videoplayseconds"] + 1.0)
    sample[["authorid", "bgm_song_id", "bgm_singer_id"]] = sample[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)
    return sample