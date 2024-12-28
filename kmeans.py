import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import plotly.express as px
import pandas as pd

# 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
def preprocess_data(x):
    # 展平图像
    x_flat = x.reshape(x.shape[0], -1)
    # 归一化
    x_flat = x_flat.astype('float32') / 255.
    # 标准化
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_flat)
    return x_scaled

# 评估聚类效果
def evaluate_clustering(x_data, y_true, y_pred):
    """评估聚类效果的多个指标"""
    # 计算轮廓系数 (-1到1，越大越好)
    silhouette = silhouette_score(x_data, y_pred)
    
    # 计算Calinski-Harabasz指数 (越大越好)
    calinski = calinski_harabasz_score(x_data, y_pred)
    
    # 计算Davies-Bouldin指数 (越小越好)
    davies = davies_bouldin_score(x_data, y_pred)
    
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

def plot_clusters_2d(x_data, labels, title='Cluster Visualization'):
    """使用PCA将数据降至2维并可视化聚类结果"""
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
        'Class': y_pred
    })
    
    fig = px.scatter_3d(df, x='Latent Dimension 1', y='Latent Dimension 2', z='Latent Dimension 3',
                        color='Class', title=title, 
                        color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.show()

# 主实验流程
def run_kmeans_experiment():
    # 预处理数据
    x_train_processed = preprocess_data(x_train)
    x_test_processed = preprocess_data(x_test)
    
    # 训练K-means
    print("开始训练K-means...")
    kmeans = KMeans(n_clusters=10, random_state=42)
    
    # 在训练集上训练和预测
    train_pred = kmeans.fit_predict(x_train_processed)
    
    # 在测试集上预测
    test_pred = kmeans.predict(x_test_processed)
    
    # 评估结果
    print("\n训练集评估:")
    train_metrics = evaluate_clustering(x_train_processed, y_train, train_pred)
    
    print("\n测试集评估:")
    test_metrics = evaluate_clustering(x_test_processed, y_test, test_pred)
    
    # 可视化结果
    print("\n绘制聚类可视化...")
    plot_latent_space_3d_interactive(x_test_processed, test_pred, 'K-means Clustering Results')
    
    return train_metrics, test_metrics

# 运行实验
if __name__ == "__main__":
    train_metrics, test_metrics = run_kmeans_experiment()