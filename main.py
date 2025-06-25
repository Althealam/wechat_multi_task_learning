# -*- coding: utf-8 -*-

import logging, json, argparse, os, data_preprocess_and_engineering, model, feature_embedding, feature, model
import tensorflow as tf
from utils import *
from feature_embedding import *
from data_preprocess_and_engineering import *
importlib.reload(model)
importlib.reload(feature)
from feature import *
from model import *
from wandb.keras import WandbCallback
from tensorflow.keras.losses import BinaryFocalCrossentropy
import wandb
from tensorflow.keras.metrics import AUC
import wandb
from wandb.keras import WandbCallback


# 训练配置
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='baseline', help='experiment name')
parser.add_argument('--running_mode', type=str, default='train', help='model running mode(train,export,predict)')
parser.add_argument('--model_path', type=str, default='/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/models', help='Model save path')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--label', type=str, default='label', help='label name')
parser.add_argument('--epoch', type=int, default=20, help='epoch.')
parser.add_argument('--max_steps', type=int, default=600000, help='Buffer size for a reader')
parser.add_argument('--eval_max_step', type=int, default=1, help='eval max step')

# 数据存储配置
# 1. 原始数据路径
parser.add_argument('--feed_info_path', type=str, default='./data/feed_info.csv', help='feed info path')
parser.add_argument('--user_action_path', type=str, default='./data/user_action.csv', help='user action path')
parser.add_argument('--feed_embeddings_path', type=str, default='./data/feed_embeddings.csv', help='feed embeddings path')
# 2. 后续要存储的数据路径
parser.add_argument('--feed_word2vec_embedding_path', type=str, default='./data/embeddings/feed_word2vec_embeddings.csv', help='feed word2vec embedding path')
parser.add_argument('--feed_deepwalk_embedding_path', type=str, default='./data/embeddings/feed_deepwalk_embeddings.csv', help='feed deepwalk embedding path')
parser.add_argument('--user_embedding_path', type=str, default='./data/embeddings/user_embeddings.csv', help='user embedding path')
parser.add_argument('--features_config_path', type=str, default='./data/config/features_config.json', help='features config path')
# 3. encoder路径
parser.add_argument('--encoder_path', type=str, default='./data/encoders', help='encoder path')
parser.add_argument('--encoder_machine_path', type=str, default='./data/encoders/encoder_machine.txt', help='encoder machine path')
parser.add_argument('--encoder_manual_path', type=str, default='./data/encoders/encoder_manual.txt', help='encoder manual path')
# 4. 用户行为序列路径
parser.add_argument('--features_path', type=str, default='./data/features', help='features path')
parser.add_argument('--user_history_sequences_path', type=str, default='./data/features/user_history_sequences.json', help='user history sequences path')
# 5. 其他路径
parser.add_argument('--config_path', type=str, default='./data/config', help='features config path')
# 6. 分步骤存储的数据
parser.add_argument('--step_1_data_preprocess_and_engineering_path', type=str, default='./data/step_1_data_preprocess_and_engineering', help='data preprocess and engineering path')

# 模型配置
parser.add_argument('--embedding_size', type=int, default=16, help='embedding_size')
parser.add_argument('--dnn_hidden_units', type=str, default='64, 16', help='dnn hidden units')
parser.add_argument('--model_name', type=str, default='base', help='model name: base/mmoe')
parser.add_argument('--embedding_dim', type=int, default=128, help='embedding size')
parser.add_argument('--act', type=str, default='relu', help='activation function')
parser.add_argument('--l2_reg', type=float, default=1e-5, help='l2 reg')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--dnn_use_bn', type=bool, default=True, help='use bn')
args = parser.parse_args()


def extract_tf_flags():
    config = vars(args)
    logging.info("===================== FLAGS =====================")
    logging.info(json.dumps(config, indent=4))
    logging.info("==================== END FLAGS ====================")
    return config


def train_test_split_tfdata(data, feature_names, batch_size=256, shuffle_buffer_size=10000):
    # 1. 划分训练集和验证集
    train_data = data[data['date_'] < 14]
    valid_data = data[data['date_'] == 14]
    print("训练集数量：", len(train_data))
    print("验证集数量：", len(valid_data))

    def process_features(df):
        features = []
        for feat in feature_names:
            col = df[feat].values
            if isinstance(col[0], list):
                maxlen = max([len(x) for x in col])
                padded = pad_sequences(col, maxlen=maxlen, padding='post', truncating='post')
                features.append(np.array(padded, dtype=np.int32))
            elif isinstance(col[0], (list, np.ndarray)):
                flattened = np.array([x[0] for x in col], dtype=np.float32)
                features.append(flattened)
            else:
                features.append(np.array(col))
        return features

    def process_labels(df):
        return {
            'click_avatar_output': np.array(df['click_avatar'].values, dtype=np.int32),
            'read_comment_output': np.array(df['read_comment'].values, dtype=np.int32),
            'like_output': np.array(df['like'].values, dtype=np.int32),
            'forward_output': np.array(df['forward'].values, dtype=np.int32),
        }

    # 2. 处理特征和标签
    train_features = process_features(train_data)
    valid_features = process_features(valid_data)
    train_labels = process_labels(train_data)
    valid_labels = process_labels(valid_data)

    # 3. 构造字典形式输入（用于配合 Keras 的 Functional API）
    def pack_features(features_list):
        return {f: v for f, v in zip(feature_names, features_list)}

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (pack_features(train_features), train_labels)
    )
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (pack_features(valid_features), valid_labels)
    )

    # 4. Shuffle, batch, prefetch
    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(tf_config['batch_size']).prefetch(tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(tf_config['batch_size']).prefetch(tf.data.AUTOTUNE)

    return train_dataset, valid_dataset


# def train_test_split(data):
#     # 划分训练集和验证集
#     train_data=data[data['date_']<14]
#     valid_data=data[data['date_']==14]
#     print("训练集数量：",len(train_data))
#     print("验证集数量：", len(valid_data))

#     train_features = [np.array(train_data[feat].values) for feat in feature_names]
#     valid_features = [np.array(valid_data[feat].values) for feat in feature_names]
#     train_labels = {
#         'click_avatar_output': np.array(train_data['click_avatar'].values, dtype=np.int32),
#         'read_comment_output': np.array(train_data['read_comment'].values, dtype=np.int32),
#         'like_output': np.array(train_data['like'].values, dtype=np.int32),
#         'forward_output': np.array(train_data['forward'].values, dtype=np.int32),
#     }
#     valid_labels = {
#         'click_avatar_output': np.array(valid_data['click_avatar'].values, dtype=np.int32),
#         'read_comment_output': np.array(valid_data['read_comment'].values, dtype=np.int32),
#         'like_output': np.array(valid_data['like'].values, dtype=np.int32),
#         'forward_output': np.array(valid_data['forward'].values, dtype=np.int32),
#     }

#     processed_train_features = []
#     for feature in train_features:
#         # 处理 list of list（如行为序列等）
#         if isinstance(feature[0], list):
#             maxlen = max([len(x) for x in feature]) 
#             padded = pad_sequences(feature, maxlen=maxlen, padding='post', truncating='post')
#             processed_train_features.append(np.array(padded, dtype=np.int32))
#         # 处理 list([0]) 类似的一维数据
#         elif isinstance(feature[0], (list, np.ndarray)):
#             flattened = np.array([x[0] for x in feature], dtype=np.float32)
#             processed_train_features.append(flattened)
#         # 正常的一维数组直接加入
#         else:
#             processed_train_features.append(np.array(feature))
    
#     processed_valid_features = []
#     for feature in valid_features:
#         # 处理 list of list（如行为序列等）
#         if isinstance(feature[0], list):
#             maxlen = max([len(x) for x in feature]) 
#             padded = pad_sequences(feature, maxlen=maxlen, padding='post', truncating='post')
#             processed_valid_features.append(np.array(padded, dtype=np.int32))
#         # 处理 list([0]) 类似的一维数据
#         elif isinstance(feature[0], (list, np.ndarray)):
#             flattened = np.array([x[0] for x in feature], dtype=np.float32)
#             processed_valid_features.append(flattened)
#         # 正常的一维数组直接加入
#         else:
#             processed_valid_features.append(np.array(feature))
    
#     return processed_train_features, processed_valid_features, train_labels, valid_labels

def main():
    # 获取模型运行配置
    tf_config = extract_tf_flags()

    # 初始化wandb 
    wandb.init(
        project = 'MMoE Project Baseline',
        name = tf_config.get('experiment_name', 'baseline'),
        config = tf_config
    )

    # 检查 GPU 状态
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU 设备已启用: {gpus}")
        except RuntimeError as e:
            print(f"GPU 配置错误: {e}")
    else:
        print("警告: 未检测到 GPU 设备")
    
    # 详细设备信息
    print("\n======= 硬件加速详情 =======")
    print("CUDA 编译版本:", tf.sysconfig.get_build_info()["cuda_version"])
    print("cuDNN 编译版本:", tf.sysconfig.get_build_info()["cudnn_version"])
    print("实际检测到的 GPU:", tf.config.list_physical_devices('GPU'))
    print("==========================\n")
    

    # 读取所需要的数据
    feed = pd.read_csv(tf_config['feed_info_path'])
    action = pd.read_csv(tf_config['user_action_path'])
    ##### 如果资源不够的话，要对这里的action的数量做限制
    # action = action.head(10000) # 限制action数量
    feed_embeddings = pd.read_csv(tf_config['feed_embeddings_path'])

    # ============= 进行数据处理和特征工程 ============
    print("######### Step1: 进行数据处理和特征工程 ###############")
    data, user_features, feed_features = preprocess_data(feed, action, tf_config)
    data = reduce_mem_usage(data)
    user_features = reduce_mem_usage(user_features)
    feed_features = reduce_mem_usage(feed_features)

    # ============= 获取feed embedding, user embedding, author embedding和用户的历史交互序列 ====================
    print("########## Step2: 获取feed embedding和user embedding以及用户历史交互序列 ###############")
    word2vec_feed_embedding, feed_embeddings = get_feed_embedding(data, feed_features, feed_embeddings)
    word2vec_feed_embedding = reduce_mem_usage(word2vec_feed_embedding)
    feed_embeddings = reduce_mem_usage(feed_embeddings)

    user_embeddings, user_history_sequences = get_user_embedding(data, word2vec_feed_embedding, tf_config)
    user_embeddings = reduce_mem_usage(user_embeddings)

    author_embeddings = get_author_embedding(data)
    author_embeddings = reduce_mem_usage(author_embeddings)


    # ============= 将历史交互序列展开为表格数据 ==============
    print("########## Step3: 处理历史交互序列 #################")
    user_history_sequences = user_history_to_dataframe(user_history_sequences) 
    user_history_sequences = reduce_mem_usage(user_history_sequences)

    # ============= 拼接user_features, feed_features, feed_embeddings, user_embeddings, user_history_sequences =============
    print("########## Step4: 拼接所有特征 ########################")
    data = pd.merge(data, user_features, on = 'userid')
    data = pd.merge(data, feed_features, on = 'feedid')
    data = pd.merge(data, word2vec_feed_embedding, on='feedid')
    data = pd.merge(data, feed_embeddings, on='feedid')
    data = pd.merge(data, user_embeddings, on='userid')
    data = pd.merge(data, author_embeddings, on='authorid')
    data = pd.merge(data, user_history_sequences, on = 'userid')
    data = reduce_mem_usage(data)

    # ============= 处理模型输入的数据格式 =================
    print('########## Step5: 开始处理模型输入的数据格式 ############')
    data_transform = model_input(data)
    data = reduce_mem_usage(data_transform)
    # data_transform.to_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/data_transform.csv', index=False)
    # print("数据处理完毕，并且成功保存！")
    
    # ============= 开始获取features_config =============
    print("########## Step6: 开始基于模型输入获取特征配置文件 #############")
    features_config = get_features_config(data, tf_config, feed, user_features)
    features_config = convert_numpy_types(features_config) # 转换类型，将所有numpy数值类型转换为python原生类型，否则存储为json时会出现报错
    os.makedirs(tf_config['config_path'], exist_ok=True)
    save_json_file(tf_config['features_config_path'], features_config)
    
    # # # ============= 开始构建模型 ================
    if tf_config.get("running_mode")=='export':
        serving_model = get_model(tf_config.get('model_name'), features_config, tf_config, word2vec_feed_embedding, user_embeddings, author_embeddings, is_training=False)
        tf.saved_model.save(serving_model, tf_config['model_path']+'/exported')
    elif tf_config.get('running_mode')=='predict': # 预测
        pass
    else: # 训练模式 
        with strategy.scope(): # 分布式训练（如果只有一张卡的话可以单卡训练，注释掉这行）
            model = get_model("base", features_config, tf_config, word2vec_feed_embedding, user_embeddings, author_embeddings)
            # processed_train_features, processed_valid_features, train_labels, valid_labels = train_test_split(data)
            train_dataset, valid_dataset = train_test_split_tfdata(
                data=data,  
                feature_names=feature_names,
                batch_size=tf_config['batch_size']
            )
            # 创建分布式策略（MirroredStrategy 会自动同步多个GPU）
            strategy = tf.distribute.MirroredStrategy()

            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
            print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

            # 编译模型
            model.compile(
                optimizer='adam',
                loss={
                    'read_comment_output': 'binary_crossentropy',
                    'like_output': 'binary_crossentropy',
                    'click_avatar_output': 'binary_crossentropy',
                    'forward_output': 'binary_crossentropy'
                },
                metrics=['AUC']
            )
            # 开始训练
            history = model.fit(
                train_dataset,
                validation_data=valid_dataset,
                epochs = tf_config['epoch'],
                callbacks = [ # 设置Wandb监控模型训练情况
                    WandbCallback(
                        monitor = 'val_loss',
                        save_weights_only=True,
                        log_evaluation = True,
                        validation_data = valid_dataset
                    )
                ]
            )


if __name__=='__main__':
    main()