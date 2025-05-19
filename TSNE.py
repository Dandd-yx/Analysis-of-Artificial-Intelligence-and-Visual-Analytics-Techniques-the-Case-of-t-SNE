import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# 加载模型
model = load_model('CIFAR10_CNN_weights_32-10.h5')

# 加载数据
(_, _), (x_test, y_test) = cifar10.load_data()

# 取随机1000张图片作为测试集
np.random.seed(42)  # 设置随机种子以保证可重复性
indices = np.random.choice(x_test.shape[0], 2500, replace=False)
x_test = x_test[indices]
y_test = y_test[indices]

# 数据预处理
x_test = x_test.astype('float32') / 255.0

# 假设我们想从倒数第二层提取特征
layer_name = 'dense'  # 将其替换为你模型中某一层的名字
intermediate_layer_model = Model(inputs=model.inputs,
                                outputs=model.get_layer(layer_name).output)
features = intermediate_layer_model.predict(x_test)

# 进行t-SNE降维
tsne = TSNE(n_components=2, random_state=42)  # 选择两个组件以方便可视化
features_2d = tsne.fit_transform(features.reshape(features.shape[0], -1))

# 将标签转换为一维数组
y_test = y_test.flatten()

plt.figure(figsize=(10, 10))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
for i in range(10):
    indices = np.where(y_test == i)
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], c=colors[i], label=str(i))
plt.legend()
plt.show()