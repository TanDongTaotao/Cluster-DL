#! -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import imageio, os
from tensorflow.keras.datasets import mnist
# from keras.datasets import fashion_mnist as mnist
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


batch_size = 100
latent_dim = 20
epochs = 100
num_classes = 10
img_dim = 28
filters = 20
intermediate_dim = 256

# 加载MNIST数据集
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, img_dim, img_dim, 1))
x_test = x_test.reshape((-1, img_dim, img_dim, 1))

# 搭建模型
x = Input(shape=(img_dim, img_dim, 1))
h = x

# 编码器卷积块
# 28x28 -> 14x14 -> 7x7
for i in range(2):
    filters *= 2
    # 第一个卷积层： 降采样
    h = Conv2D(filters=filters, kernel_size=3, strides=2, padding='same')(h)
    h = LeakyReLU(0.2)(h)
    # 第二个卷积层： 特征提取
    h = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(h)
    h = LeakyReLU(0.2)(h)

h_shape = K.int_shape(h)[1:]
h = Flatten()(h)
z_mean = Dense(latent_dim)(h)  # p(z|x)的均值
z_log_var = Dense(latent_dim)(h)  # p(z|x)的方差

encoder = Model(x, z_mean)  # 通常认为z_mean就是所需的隐变量编码

z = Input(shape=(latent_dim,))
h = z
h = Dense(np.prod(h_shape))(h)
h = Reshape(h_shape)(h)

# 解码器卷积转置块
# 7x7 -> 14x14 -> 28x28
for i in range(2):
    # 第一个卷积转置层：特征处理
    h = Conv2DTranspose(filters=filters, kernel_size=3, strides=1, padding='same')(h)
    h = LeakyReLU(0.2)(h)
    # 第二个卷积转置层：上采样
    h = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(h)
    h = LeakyReLU(0.2)(h)
    filters //= 2
x_recon = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same')(h)

decoder = Model(z, x_recon)  # 解码器
generator = decoder

z = Input(shape=(latent_dim,))
y = Dense(intermediate_dim, activation='relu')(z)
y = Dense(num_classes, activation='softmax')(y)

classfier = Model(z, y)  # 隐变量分类器

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
x_recon = decoder(z)
y = classfier(z)

class Gaussian(Layer):
    """这是个简单的层,定义q(z|y)中的均值参数，每个类别配一个均值。
    然后输出“z - 均值”,为后面计算loss准备。
    """
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)
    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean', shape=(self.num_classes, latent_dim), initializer='zeros')
    def call(self, inputs):
        z = inputs  # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z - K.expand_dims(self.mean, 0)
    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])

def plot_latent_space(x_encoded, y_true, title='Latent Space'):
    """绘制隐空间的点图以可视化聚类性能"""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_encoded[:, 0], x_encoded[:, 1], c=y_true, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(num_classes))
    plt.title(title)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True)
    plt.show()

def plot_latent_space_3d_interactive(x_encoded, y_true, title='3D Latent Space'):
    """使用Plotly绘制交互式3D点图以可视化聚类性能"""
    df = pd.DataFrame({
        'Latent Dimension 1': x_encoded[:, 0],
        'Latent Dimension 2': x_encoded[:, 1],
        'Latent Dimension 3': x_encoded[:, 2],
        'Class': y_true
    })
    
    fig = px.scatter_3d(df, x='Latent Dimension 1', y='Latent Dimension 2', z='Latent Dimension 3',
                        color='Class', title=title, color_continuous_scale='Viridis')
    
    fig.show()

def evaluate_clustering(x_encoded, y_true, y_pred):
    """评估聚类效果的多个指标"""
    # 计算轮廓系数 (-1到1，越大越好)
    silhouette = silhouette_score(x_encoded, y_pred)
    
    # 计算Calinski-Harabasz指数 (越大越好)
    calinski = calinski_harabasz_score(x_encoded, y_pred)
    
    # 计算Davies-Bouldin指数 (越小越好)
    davies = davies_bouldin_score(x_encoded, y_pred)
    
    # 计算标准化互信息 (0到1，越大越好)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    
    # 计算调整兰德指数 (-1到1，越大越好)
    ari = adjusted_rand_score(y_true, y_pred)
    
    # 打印评估结果
    print("\n聚类评估指标:")
    print(f"轮廓系数 (Silhouette): {silhouette:.4f}")
    print(f"Calinski-Harabasz指数: {calinski:.4f}")
    print(f"Davies-Bouldin指数: {davies:.4f}")
    print(f"标准化互信息 (NMI): {nmi:.4f}")
    print(f"调整兰德指数 (ARI): {ari:.4f}")
    
    return silhouette, calinski, davies, nmi, ari

gaussian = Gaussian(num_classes)
z_prior_mean = gaussian(z)

# 建立模型
vae = Model(x, [x_recon, z_prior_mean, y])
# 保存模型架构图
plot_model(vae, to_file='vae_model.png', show_shapes=True, show_layer_names=True)

# 下面一大通都是为了定义loss
z_mean = K.expand_dims(z_mean, 1)
z_log_var = K.expand_dims(z_log_var, 1)

lamb = 2.5  # 这是重构误差的权重，它的相反数就是重构方差，越大意味着方差越小。
xent_loss = 0.5 * K.mean((x - x_recon)**2, 0)
kl_loss = - 0.5 * (z_log_var - K.square(z_prior_mean))
kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0)
cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)
vae_loss = lamb * K.sum(xent_loss) + K.sum(kl_loss) + K.sum(cat_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置GPU内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"使用GPU: {gpus}")
    except RuntimeError as e:
        print(e)

vae.fit(x_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

means = K.eval(gaussian.mean)
x_train_encoded = encoder.predict(x_train)
y_train_pred = classfier.predict(x_train_encoded).argmax(axis=1)
x_test_encoded = encoder.predict(x_test)
y_test_pred = classfier.predict(x_test_encoded).argmax(axis=1)

# 在训练后评估训练集和测试集的聚类效果
print("\n训练集评估:")
train_metrics = evaluate_clustering(x_train_encoded, y_train_, y_train_pred)

print("\n测试集评估:")
test_metrics = evaluate_clustering(x_test_encoded, y_test_, y_test_pred)

# # 3维展示
# from sklearn.decomposition import PCA  # 添加PCA导入
# # 获取编码后的潜在空间表示
# x_test_encoded = encoder.predict(x_test)
# # 使用PCA降到3维
# pca = PCA(n_components=3)
# x_test_encoded_3d = pca.fit_transform(x_test_encoded)
# plot_latent_space_3d_interactive(x_test_encoded_3d, y_test_, title='Test Latent Space 3D')

# 二维展示
from sklearn.decomposition import PCA  # 添加PCA导入
# 获取编码后的潜在空间表示
x_test_encoded = encoder.predict(x_test)
# 使用PCA降到2维
pca = PCA(n_components=2)
x_test_encoded_2d = pca.fit_transform(x_test_encoded)

# 绘制降维后的潜在空间
plot_latent_space(x_test_encoded_2d, y_test_pred, title='Test Latent Space (PCA)')


def cluster_sample(path, category=0):
    """观察被模型聚为同一类的样本
    """
    n = 8
    figure = np.zeros((img_dim * n, img_dim * n))
    idxs = np.where(y_train_pred == category)[0]
    for i in range(n):
        for j in range(n):
            digit = x_train[np.random.choice(idxs)]
            digit = digit.reshape((img_dim, img_dim))
            figure[i * img_dim: (i + 1) * img_dim, j * img_dim: (j + 1) * img_dim] = digit
    # 将图像数据转换为8位整数
    imageio.imwrite(path, (figure * 255).astype(np.uint8))

def random_sample(path, category=0, std=1):
    """按照聚类结果进行条件随机生成
    """
    n = 8
    figure = np.zeros((img_dim * n, img_dim * n))
    for i in range(n):
        for j in range(n):
            noise_shape = (1, latent_dim)
            z_sample = np.array(np.random.randn(*noise_shape)) * std + means[category]
            x_recon = generator.predict(z_sample)
            digit = x_recon[0].reshape((img_dim, img_dim))
            figure[i * img_dim: (i + 1) * img_dim, j * img_dim: (j + 1) * img_dim] = digit
    # 将图像数据转换为8位整数
    imageio.imwrite(path, (figure * 255).astype(np.uint8))

if not os.path.exists('samples'):
    os.mkdir('samples')

for i in range(10):
    cluster_sample(f'samples/聚类类别_{i}.png', i)
    random_sample(f'samples/类别生成_{i}.png', i)

right = 0.
for i in range(10):
    _ = np.bincount(y_train_[y_train_pred == i])
    right += _.max()

print(f'train acc: {right / len(y_train_)}')

right = 0.
for i in range(10):
    _ = np.bincount(y_test_[y_test_pred == i])
    right += _.max()

print(f'test acc: {right / len(y_test_)}')