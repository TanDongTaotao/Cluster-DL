import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import plotly.express as px
import pandas as pd

# 参数设置
n_components = 20  # 与VAE和AE相同的潜在空间维度
n_clusters = 10

def preprocess_data(x):
    # 展平图像
    x_flat = x.reshape(x.shape[0], -1)
    # 归一化
    x_flat = x_flat.astype('float32') / 255.
    # 标准化
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_flat)
    return x_scaled

def evaluate_clustering(x_data, y_true, y_pred):
    """评估聚类效果的多个指标"""
    silhouette = silhouette_score(x_data, y_pred)
    calinski = calinski_harabasz_score(x_data, y_pred)
    davies = davies_bouldin_score(x_data, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    print("\n聚类评估指标:")
    print(f"轮廓系数 (Silhouette): {silhouette:.4f}")
    print(f"Calinski-Harabasz指数: {calinski:.4f}")
    print(f"Davies-Bouldin指数: {davies:.4f}")
    print(f"标准化互信息 (NMI): {nmi:.4f}")
    print(f"调整兰德指数 (ARI): {ari:.4f}")
    
    return silhouette, calinski, davies, nmi, ari

def plot_clusters_2d(x_data, labels, title='Cluster Visualization'):
    """使用PCA将数据降至2维并可视化聚类结果"""
    pca_viz = PCA(n_components=2)
    x_pca = pca_viz.fit_transform(x_data)
    
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
        'Class': y_pred.astype(str)
    })
    
    fig = px.scatter_3d(df, x='Latent Dimension 1', y='Latent Dimension 2', z='Latent Dimension 3',
                        color='Class', title=title, 
                        color_discrete_sequence=px.colors.qualitative.Set1)
    
    fig.show()

def run_pca_kmeans_experiment():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 预处理数据
    print("预处理数据...")
    x_train_processed = preprocess_data(x_train)
    x_test_processed = preprocess_data(x_test)
    
    # PCA降维
    print(f"\n使用PCA降维到{n_components}维...")
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train_processed)
    x_test_pca = pca.transform(x_test_processed)
    
    # 输出解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    print(f"累计解释方差比: {cumulative_variance_ratio[-1]:.4f}")
    
    # 绘制解释方差比曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_components + 1), cumulative_variance_ratio, 'bo-')
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差比')
    plt.title('PCA累计解释方差比')
    plt.grid(True)
    plt.show()
    
    # K-means聚类
    print("\n执行K-means聚类...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    train_pred = kmeans.fit_predict(x_train_pca)
    test_pred = kmeans.predict(x_test_pca)
    
    # 评估结果
    print("\n训练集评估:")
    train_metrics = evaluate_clustering(x_train_pca, y_train, train_pred)
    
    print("\n测试集评估:")
    test_metrics = evaluate_clustering(x_test_pca, y_test, test_pred)
    
    # 可视化结果
    print("\n绘制聚类可视化...")
    plot_latent_space_3d_interactive(x_test_pca, test_pred, 'PCA + K-means Clustering Results')
    
    # 计算重构误差
    x_train_recon = pca.inverse_transform(x_train_pca)
    x_test_recon = pca.inverse_transform(x_test_pca)
    train_mse = np.mean((x_train_processed - x_train_recon) ** 2)
    test_mse = np.mean((x_test_processed - x_test_recon) ** 2)
    
    print("\n重构误差:")
    print(f"训练集 MSE: {train_mse:.6f}")
    print(f"测试集 MSE: {test_mse:.6f}")
    
    return train_metrics, test_metrics, cumulative_variance_ratio[-1]

if __name__ == "__main__":
    train_metrics, test_metrics, variance_ratio = run_pca_kmeans_experiment()
