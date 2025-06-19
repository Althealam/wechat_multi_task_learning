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
parser.add_argument('--feed_deepwalk_embedding_path', type=str, default='/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/embeddings/deepwalk_feed_embedding.csv', help='feed deepwalk embedding path')
parser.add_argument('--user_embedding_path', type=str, default='/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/embeddings/user_embedding.csv', help='user embedding path')
parser.add_argument('--features_config_path', type=str, default='/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/config/features_config.json', help='features config path')
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

# 读取特征配置
# with open('/opt/tiger/toutiao_search_rec_pyspark_code/nn_model/config/embedding_feat_dict.json', 'r') as f:
#     features_config = json.load(f)

# features_config = generate_tfrecord_config(embedding_feat_dict, feature_names, label_name)

def extract_tf_flags():
    config = vars(args)
    logging.info("===================== FLAGS =====================")
    logging.info(json.dumps(config, indent=4))
    logging.info("==================== END FLAGS ====================")
    return config

def main():
    # 获取模型运行配置
    tf_config = extract_tf_flags()

    # 读取所需要的数据
    feed = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/feed_info.csv')
    action = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/user_action.csv')
    feed_embeddings = pd.read_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/feed_embeddings.csv')

    # ============= 进行数据处理和特征工程 ============
    print("######### Step1: 进行数据处理和特征工程 ###############")
    data, user_features, feed_features = preprocess_data(feed, action)

    print("正在进行内存优化...")
    data = reduce_mem_usage(data)
    user_features = reduce_mem_usage(user_features)
    feed_features = reduce_mem_usage(feed_features)
    print("内存优化完毕...")

    # print("正在存储经过数据处理和特征工程的数据...")
    # processes = []
    # os.makedirs('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/features', exist_ok=True)
    # processes.append(multiprocessing.Process(target=save_csv, args=(feed_features, '/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/features/feed_features.csv')))
    # processes.append(multiprocessing.Process(target=save_csv, args=(user_features, '/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/features/user_features.csv')))
    # processes.append(multiprocessing.Process(target=save_csv, args=(data, '/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/data.csv')))
    # # 启动所有进程
    # for p in processes:
    #     p.start()
    # # 等待所有进程完成
    # for p in processes:
    #     p.join()
    # print("数据存储成功！")

    # ============= 获取feed embedding, user embedding和用户的历史交互序列 ====================
    print("########## Step2: 获取feed embedding和user embedding以及用户历史交互序列 ###############")
    deepwalk_feed_embedding, feed_embeddings = get_feed_embedding(data, feed_features, feed_embeddings)

    print("正在进行内存优化...")
    deepwalk_feed_embedding = reduce_mem_usage(deepwalk_feed_embedding)
    feed_embeddings = reduce_mem_usage(feed_embeddings)
    print("内存优化完毕...")

    # print("正在存储feed embeddings数据...")
    # processes = []
    # os.makedirs('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/embeddings', exist_ok=True)
    # processes.append(multiprocessing.Process(target=save_csv, args=(deepwalk_feed_embedding, '/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/embeddings/deepwalk_feed_embedding.csv')))
    # processes.append(multiprocessing.Process(target=save_csv, args=(feed_embeddings, '/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/embeddings/deepwalk_feed_embedding.csv')))
    # # 启动所有进程
    # for p in processes:
    #     p.start()
    # # 等待所有进程完成
    # for p in processes:
    #     p.join()
    # print("数据存储成功！")


    print("====获取user embedding====")
    user_embeddings, user_history_sequences = get_user_embedding(data, deepwalk_feed_embedding)

    print("正在进行内存优化...")
    user_embeddings = reduce_mem_usage(user_embeddings)
    # user_history_sequences = reduce_mem_usage(user_history_sequences) # json文件无法进行内存优化
    print("内存优化完毕...")
    # print("正在存储user embeddings数据...")
    # processes = []
    # processes.append(multiprocessing.Process(target=save_csv, args=(user_embeddings, '/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/embeddings/user_embedding.csv')))
    # # 启动所有进程
    # for p in processes:
    #     p.start()
    # # 等待所有进程完成
    # for p in processes:
    #     p.join()
    # print("数据存储成功！")

    # ============= 将历史交互序列展开为表格数据 ==============
    print("########## Step3: 处理历史交互序列 #################")
    user_history_sequences = user_history_to_dataframe(user_history_sequences) 
    user_history_sequences = reduce_mem_usage(user_history_sequences)

    # ============= 拼接user_features, feed_features, feed_embeddings, user_embeddings, user_history_sequences =============
    print("########## Step4: 拼接所有特征 ########################")
    data = pd.merge(data, user_features, on = 'userid')
    data = pd.merge(data, feed_features, on = 'feedid')
    data = pd.merge(data, feed_embeddings, on='feedid')
    data = pd.merge(data, user_embeddings, on='userid')
    data = pd.merge(data, user_history_sequences, on = 'userid')
    data = reduce_mem_usage(data)

    # print("正在存储拼接了所有特征和embeddings的数据...")
    # processes = []
    # processes.append(multiprocessing.Process(target=save_csv, args=(data, '/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/data_with_features_embedding.csv')))
    # # 启动所有进程
    # for p in processes:
    #     p.start()

    # # 等待所有进程完成
    # for p in processes:
    #     p.join()

    # ============= 处理模型输入的数据格式 =================
    print('########## Step5: 开始处理模型输入的数据格式 ############')
    data_transform = model_input(data)
    data = reduce_mem_usage(data)
    # data_transform.to_csv('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/data_transform.csv', index=False)
    # print("数据处理完毕，并且成功保存！")
    
    # ============= 开始获取features_config =============
    print("########## Step6: 开始基于模型输入获取特征配置文件 #############")
    features_config = get_features_config(data_transform)
    features_config = convert_numpy_types(features_config) # 转换类型，将所有numpy数值类型转换为python原生类型，否则存储为json时会出现报错
    os.makedirs('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/config', exist_ok=True)
    save_json_file('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/config/features_config.json', features_config)
    print("特征配置获取完毕，并且成功保存！")
    
    # # ============= 开始构建模型 ================
    # features_config = read_json_file('/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/data/config/features_config.json')
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