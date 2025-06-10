## 项目简介
该项目数据来自2021微信大数据挑战赛的数据集。
本项目主要是为了学习多目标建模和TF的常见训练方式（主要是在实习的过程中发现对TF的熟练度不够T_T，所以特地弄一个项目来学习）

## 项目结构
- data_preprocess_and_data_engineering：数据预处理和数据工程
- models：模型代码
- layers：自定义的层
- model_config：模型参数
- utils：自定义的工具函数
- MMoE.ipynb：MMoE模型的训练代码
- features.py：模型使用的特征


## 更新记录
2025/5/27:
- 跑通PLE和MMoE，目前还存在报错需要排查

2025/6/3:
- 预处理数据，包含填充缺失值，处理序列特征为数组（按照分号和空格进行划分）
- TODO：获取embedding_feat_dict 将数据处理干净后划分为train, test, val并存储起来

2025/6/5:
- 跑通MMoE模型，目前来看效果还不错
- TODO：优化模型结构，弄懂各个参数，把项目结构完善一下，生成一个完整的README文件

2025/6/7
- 优化项目结构

2025/6/9
- 增加description（序列特征），可以用来考虑常见的DIN等处理序列特征的方式
- 备注：
* 加入了description后第一轮训练的AUC反而有所下降，可能是因为description很多都是噪声，或者是对description做下降，可能是因为d
escription很多都是噪声，或者是对description做平均池化的效果一般
* 在训练到第二个epoch的时候出现了跷跷板现象，也就是read_comment的AUC在涨，但是其他的click_avator, forward, like的AUC在下降，需要寻找解决方法
- TODO：
* 优化模型结构：对description这个序列特征使用GRU+Attention（DIEN）建模？
* 解决跷跷板现象？

2025/6/10
- 优化模型结构：对description使用GRU+Attention进行embedding处理，其他的sequence_feature还是使用平均池化
- TODO：创建一个新分支，处理成tf常见的训练模式（刚好学习一下tf的训练，数据弄成tfrecord而不是csv）