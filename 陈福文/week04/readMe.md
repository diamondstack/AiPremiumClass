作业内容：
1. 搭建的神经网络，使用olivettiface数据集进行训练。
2. 结合归一化和正则化来优化网络模型结构，观察对比loss结果。
3. 尝试不同optimizer对模型进行训练，观察对比loss结果。


归一化：加速模型收敛并减少过拟合，稳定训练过程。重新排队。一开始队形还很好，后面人越来越多，就会出现队伍混乱的情况。
步骤：1、计算均值，2、计算方差，3、归一化，4、缩放和平移
除batchNorm外，还有以下归一化操作运算
    LayerNorm：适合长序列数据
    GroupNorm：平衡BN和LN，适合中等批量场景
    SwitchableNorm：自适应选择归一化方式，适合动态批量场景
归一化也是可以训练参数的，不可随意变更。

正则化：防止模型的过拟合，提升模型的泛化能力。
L1正则化：L1范数，权重绝对值之和
L2正则化：L2范数，权重平方和的平方根
Dropout：随机丢弃神经元，防止过拟合，提升模型的泛化能力。
熟悉就是模型的过拟合，下意识的反应。
神经网络中，每次切换临时的输出，防止过拟合。

train()
eval()
zero_grad()

优化器optimizer：更新神经网络参数的工具。根据损失函数和优化算法，自动调整参数以最小化损失。梯度*学习率
围绕梯度更新

了解优化器的实现和机制

sgd : 学习率，参数集，输出，权重衰减，动量，动量的衰减率

1、训练多少轮
2、计算梯度
3、考虑权重衰减，避免对训练数据产生记忆
4、动量更新，惯性