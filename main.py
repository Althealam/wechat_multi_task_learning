import logging, json, argparse, os, data_preprocess_and_engineering, models, feature_utils, feature_embedding
import tensorflow as tf
from utils import *
from feature_utils import *
from feature_embedding import *
from data_preprocess_and_engineering import *
importlib.reload(models)
from wandb.keras import WandbCallback
from tensorflow.keras.losses import BinaryFocalCrossentropy
import wandb

# 训练配置
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/root/repo/Wechat_Multi_Task_Learning_Recommendation_Project/models', help='Model save path')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--label', type=str, default='label', help='label name')
parser.add_argument('--epoch', type=int, default=2, help='epoch.')
parser.add_argument('--max_steps', type=int, default=600000, help='Buffer size for a reader')
parser.add_argument('--eval_max_step', type=int, default=1, help='eval max step')
# 模型配置
parser.add_argument('--embedding_size', type=int, default=16, help='embedding_size')
parser.add_argument('--dnn_hidden_units', type=str, default='64, 16', help='dnn hidden units')
parser.add_argument('--running_mode', type=str, default='train', help='train,predict,export')
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

    # ============= 进行数据处理和特征工程 ============
    print("######### Step1: 进行数据处理和特征工程 ###############")
    data, user_features, feed_features = preprocess_data()

    # ============= 获取feed embedding, user embedding和用户的历史交互序列 ====================
    print("########## Step2: 获取feed embedding和user embedding以及用户历史交互序列 ###############")
    deepwalk_feed_embedding, feed_embeddings = get_feed_embedding(data, feed_features)
    user_embeddings, user_history_sequences = get_user_embedding()

    # ============= 将历史交互序列展开为表格数据 ==============
    print("########## Step3: 处理历史交互序列 #################")
    user_history_sequences = user_history_to_dataframe(user_history_sequences) 

    # ============= 拼接user_features, feed_features, feed_embeddings, user_embeddings, user_history_sequences =============
    print("########## Step4: 拼接所有特征 ########################")
    data = pd.merge(data, user_features, on = 'userid')
    data = pd.merge(data, feed_features, on = 'feedid')
    data = pd.merge(data, feed_embeddings, on='feedid')
    data = pd.merge(data, user_embeddings, on='userid')
    data = pd.merge(data, user_history_sequences, on = 'userid')
    data.to_csv('data_with_features_embedding.csv', index=False)


if __name__=='__main__':
    main()