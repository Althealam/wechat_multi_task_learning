## 项目简介
该项目数据来自2021微信大数据挑战赛的数据集。
本项目主要是为了学习多任务学习，包括MMoE和PLE

## 运行配置
* 计算资源：两张V100 8核CPU 32G内存 + 8核CPU 16G内存
* 运行环境：Python 3.10

## 收获 
1. 项目工程化
2. 熟悉TF2.x的完整流程，包括使用tf.datasets而不是dataframe，同时使用parsers.add_argument进行命令行交互
3. 多任务学习的实现，包括MMoE
4. 完整的项目流程
5. 分布式训练
6. 使用Wandb进行模型监控而不是TensorBoard
7. 使用CPU多进程数据处理

## 更新记录
2025/6/14:
- 数据预处理：填充缺失值，播放时长离散化
- 特征工程：多值id进行tf-idf和降维，生成统计特征

2025/6/16:
- 修改DeepWalk代码，避免报错
- TODO：DeepWalk模型训练

2025/6/17
- 跑通DeepWalk代码，梳理多模态表征和DeepWalk表征的融合思路
- TODO：发现有很多交互历史的feed没有多模态表征，首先排查是否是数据的问题（数据过大，上传有问题？），其次考虑冷启动视频的处理（只有多模态表征，没有DeepWalk表征）

2025/6/18
- 获取用户的历史行为序列
- 已确认是数据的问题，多模态embedding应该包含所有feed的，但是目前的数据没有
- 修改报错，拼接user_features, feed_features, user_embedding, feed_embedding, 用户历史行为序列到data中
- TODO：数据预处理（填充数组、离散化等），构建模型

2025/6/19
- 修改DeepWalk为Word2Vec
- 重新下载并导入完整的数据集 feed大概有7w多条 符合预期
- 生成authorid的embedding
- 处理数据为模型的输入并获取features_config，并构建模型的input layers和embedding layers
- TODO：（1）融合feed的多模态embedding和word2vec embedding（2）构建完整模型

2025/6/24
- 构建模型

2025/6/25
- 构建模型：已实现通过DIN提取用户历史行为序列和目标向量之间的注意力权重，同时实现MMoE的构建
- 修正DIN Attention报错，跑通模型训练代码
- 修改数据集为tf.datasets，使用Tesla-V100-SXM2-32GB进行分布式训练，CPU8核，MEM32G，CPU2卡
- TODO：增加Tensorboard或者wandb监控模型训练情况；解决爆内存的问题