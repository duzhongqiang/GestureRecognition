# -- coding:utf-8 --
import matplotlib.pyplot as plt
import numpy as np
#优化函数loss
# x = np.arange(0, 11, 1)
# y1 = [1.3119,0.3465,0.1889,0.1258,0.0753,0.0500,0.0343,0.0234,0.0120,0.0070,0.0052] #4500sgd
# y2 = [1.0192,0.4252,0.3103,0.2592,0.2232,0.2137,0.1527,0.1317,0.1169,0.0640,0.0249] #Adam
#优化函数ACC
# x = np.arange(0, 11, 1)
# y1 = [0.5514,0.8959,0.9405,0.9611,0.9784,0.9843,0.9898,0.9930,0.9968,0.9986,0.9993]
# y2 = [0.6653,0.8573,0.9000,0.9203,0.9313,0.9353,0.9550,0.9563,0.9623,0.9820,0.9930]
#预训练模型loss
# x = np.arange(0, 9, 1)
# y1 = [1.4672, 0.4249, 0.2486,0.1651,0.1240,0.1047,0.0862,0.0741,0.0711] #nopre
# y2 = [1.2382, 0.2904, 0.1551,0.0835,0.0637,0.0430,0.0301,0.0208,0.0184] #pre
#预训练模型ACC
x = np.arange(0, 9, 1)
y1 = [0.4778,0.8645,0.9228,0.9515,0.9615,0.9683,0.9733,0.9780,0.9775]
y2 = [0.5815,0.9125,0.9515,0.9755,0.9843,0.9880,0.9900,0.9950,0.9945]
plt.plot(x, y1, x, y2)
legends = ['No pre-training', 'pre-trained']
# legends = ['SGD', 'Adam']
plt.legend(legends)
plt.grid()  # 生成网格
plt.xticks(np.arange(0, 9, 1))
plt.xlabel('epoch')
plt.ylabel('Accuracy')
# plt.rcParams['font.sans-serif']=['SimHei']
plt.title('Whether to use a pre-trained model')
# plt.title('The difference between different loss functions')
plt.show()