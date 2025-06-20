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


# 训练配置
parser = argparse.ArgumentParser()
parser.add_argument('--running_mode', type=str, default='train', help='model running mode(train,export,predict)')
parser.add_argument('--model_path', type=str, default='/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/models', help='Model save path')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--label', type=str, default='label', help='label name')
parser.add_argument('--epoch', type=int, default=2, help='epoch.')
parser.add_argument('--max_steps', type=int, default=600000, help='Buffer size for a reader')
parser.add_argument('--eval_max_step', type=int, default=1, help='eval max step')

# 数据存储配置
# 1. 原始数据路径
parser.add_argument('--feed_info_path', type=str, default='~/repo/mtl_notebook/data/feed_info.csv', help='feed info path')
parser.add_argument('--user_action_path', type=str, default='~/repo/mtl_notebook/data/user_action.csv', help='user action path')
parser.add_argument('--feed_embeddings_path', type=str, default='~/repo/mtl_notebook/data/feed_embeddings.csv', help='feed embeddings path')
# 2. 后续要存储的数据路径
parser.add_argument('--feed_word2vec_embedding_path', type=str, default='~/repo/mtl_notebook/data/embeddings/feed_word2vec_embeddings.csv', help='feed word2vec embedding path')
parser.add_argument('--feed_deepwalk_embedding_path', type=str, default='~/repo/mtl_notebook/data/embeddings/feed_deepwalk_embeddings.csv', help='feed deepwalk embedding path')
parser.add_argument('--user_embedding_path', type=str, default='~/repo/mtl_notebook/data/embeddings/user_embeddings.csv', help='user embedding path')
parser.add_argument('--features_config_path', type=str, default='~/repo/mtl_notebook/data/config/features_config.json', help='features config path')
# 3. encoder路径
parser.add_argument('--encoder_path', type=str, default='~/repo/mtl_notebook/data/encoders', help='encoder path')
parser.add_argument('--encoder_machine_path', type=str, default='~/repo/mtl_notebook/data/encoders/encoder_machine.txt', help='encoder machine path')
parser.add_argument('--encoder_manual_path', type=str, default='~/repo/mtl_notebook/data/encoders/encoder_manual.txt', help='encoder manual path')
# 4. 用户行为序列路径
parser.add_argument('--features_path', type=str, default='~/repo/mtl_notebook/data/features', help='features path')
parser.add_argument('--user_history_sequences_path', type=str, default='~/repo/mtl_notebook/data/features/user_history_sequences.json', help='user history sequences path')
# 5. 其他路径
parser.add_argument('--config_path', type=str, default='~/repo/mtl_notebook/data/config', help='features config path')
# 6. 分步骤存储的数据
parser.add_argument('--step_1_data_preprocess_and_engineering_path', type=str, default='~/repo/mtl_notebook/data/step_1_data_preprocess_and_engineering', help='data preprocess and engineering path')

# 模型配置
parser.add_argument('--embedding_size', type=int, default=16, help='embedding_size')
parser.add_argument('--dnn_hidden_units', type=str, default='64, 16', help='dnn hidden units')
parser.add_argument('--model_name', type=str, default='base', help='model name: base/mmoe')
parser.add_argument('--embedding_dim', type=int, default=16, help='embedding size')
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

def main():
    # 获取模型运行配置
    tf_config = extract_tf_flags()

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
    features_config = get_features_config(data_transform)
    features_config = convert_numpy_types(features_config) # 转换类型，将所有numpy数值类型转换为python原生类型，否则存储为json时会出现报错
    os.makedirs(tf_config['config_path'], exist_ok=True)
    save_json_file(tf_config['features_config_path'], features_config)
    
    # # # ============= 开始构建模型 ================
    # build_base_model(features_config, tf_config)
    # if tf_config.get("running_mode")=='export':
    #     serving_model = get_model(tf_config.get('model_name'), features_config, tf_config, is_training=False)
    #     tf.saved_model.save(serving_model, tf_config['model_path']+'/exported')
    # elif tf_config.get('running_mode')=='predict':
    #     pass
    # else:
    #     model = get_model("base", features_config, tf_config)
    #     model.summary()


if __name__=='__main__':
    main()