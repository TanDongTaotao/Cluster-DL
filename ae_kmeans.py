import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import plotly.express as px
import pandas as pd

# 参数设置
latent_dim = 20  # 与VAE相同的潜在空间维度
input_shape = (28, 28, 1)
batch_size = 128
epochs = 50

def build_autoencoder():
    # 编码器
    inputs = Input(shape=input_shape)
    x = Flatten()(inputs)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    encoded = Dense(latent_dim, activation='relu', name='encoded')(x)
    
    # 解码器
    x = Dense(256, activation='relu')(encoded)
    x = Dense(512, activation='relu')(x)
    x = Dense(np.prod(input_shape), activation='sigmoid')(x)
    decoded = Reshape(input_shape)(x)
    
    # 构建模型
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def preprocess_data(x):
    x = x.astype('float32') / 255.
    x = x.reshape((-1,) + input_shape)
    return x

def evaluate_clustering(x_data, y_true, y_pred):
    """评估聚类效果的多个指标"""
    silhouette = silhouette_score(x_data, y_pred)
    calinski = calinski_harabasz_score(x_data, y_pred)
    davies = davies_bouldin_score(x_data, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print("\n聚类评估指标:")
    print(f"��廓系数 (Silhouette): {silhouette:.4f}")
    print(f"Calinski-Harabasz指数: {calinski:.4f}")
    print(f"Davies-Bouldin指数: {davies:.4f}")
    print(f"标准化互信息 (NMI): {nmi:.4f}")
    print(f"调整兰德指数 (ARI): {ari:.4f}")
    
    return silhouette, calinski, davies, nmi, ari

def plot_clusters_2d(x_data, labels, title='Cluster Visualization'):
    """使用PCA将数据降至2维并可视化聚类结果"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_data)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True)
    plt.show()

def plot_latent_space_3d_interactive(x_encoded, y_pred, title='3D Latent Space'):
    """使用Plotly绘制交互式3D点图以可视化聚类性能"""
    # 使用PCA降到3维
    pca = PCA(n_components=3)
    x_encoded_3d = pca.fit_transform(x_encoded)
    
    df = pd.DataFrame({
        'Latent Dimension 1': x_encoded_3d[:, 0],
        'Latent Dimension 2': x_encoded_3d[:, 1],
        'Latent Dimension 3': x_encoded_3d[:, 2],
        'Class': y_pred.astype(str)  # 将类别转换为字符串
    })
    
    fig = px.scatter_3d(df, x='Latent Dimension 1', y='Latent Dimension 2', z='Latent Dimension 3',
                        color='Class', title=title, 
                        color_discrete_sequence=px.colors.qualitative.Set1)  # 使用Set1离散颜色序列
    
    fig.show()

def run_ae_kmeans_experiment():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 预处理数据
    x_train_processed = preprocess_data(x_train)
    x_test_processed = preprocess_data(x_test)
    
    # 构建和训练自编码器
    print("训练自编码器...")
    autoencoder, encoder = build_autoencoder()
    autoencoder.fit(x_train_processed, x_train_processed,
                   epochs=epochs,
                   batch_size=batch_size,
                   shuffle=True,
                   validation_data=(x_test_processed, x_test_processed),
                   verbose=1)
    
    # 使用编码器获取潜在空间表示
    print("\n提取潜在特征...")
    x_train_encoded = encoder.predict(x_train_processed)
    x_test_encoded = encoder.predict(x_test_processed)
    
    # 应用K-means聚类
    print("\n执行K-means聚类...")
    kmeans = KMeans(n_clusters=10, random_state=42)
    train_pred = kmeans.fit_predict(x_train_encoded)
    test_pred = kmeans.predict(x_test_encoded)
    
    # 评估结果
    print("\n训练集评估:")
    train_metrics = evaluate_clustering(x_train_encoded, y_train, train_pred)
    
    print("\n测试集评估:")
    test_metrics = evaluate_clustering(x_test_encoded, y_test, test_pred)
    
    # 可视化结果
    print("\n绘制聚类可视化...")
    plot_latent_space_3d_interactive(x_test_encoded, test_pred, 'AE + K-means Clustering Results')
    
    # 计算重构误差
    train_recon = autoencoder.predict(x_train_processed)
    test_recon = autoencoder.predict(x_test_processed)
    train_mse = np.mean((x_train_processed - train_recon) ** 2)
    test_mse = np.mean((x_test_processed - test_recon) ** 2)
    
    print("\n重构误差:")
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"测试集 MSE: {test_mse:.6f}")
    
    return train_metrics, test_metrics

if __name__ == "__main__":
    train_metrics, test_metrics = run_ae_kmeans_experiment()
