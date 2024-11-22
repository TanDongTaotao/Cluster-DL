import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from time import time
import metrics
from ConvAE import CAE
import os

class ClusteringLayer(nn.Module):
    """
    将输入样本转换为软标签的聚类层。使用t分布计算样本属于每个簇的概率。
    """
    def __init__(self, n_clusters, n_features):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        # 初始化聚类中心
        self.weight = nn.Parameter(torch.Tensor(n_clusters, n_features))
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x):
        """ t-分布计算概率
        Args:
            x: 输入数据, shape=(batch_size, n_features)
        Returns:
            q: t-分布概率, shape=(batch_size, n_clusters)
        """
        # 确保输入张量形状正确
        if len(x.shape) > 2:
            # 如果是卷积特征图，将其展平
            x = x.view(x.size(0), -1)
        
        # 计算每个样本到聚类中心的距离
        x_norm = torch.sum(x**2, dim=1, keepdim=True)
        w_norm = torch.sum(self.weight**2, dim=1, keepdim=True)
        dist = x_norm + w_norm.t() - 2 * torch.matmul(x, self.weight.t())
        
        # 计算 Student's t-distribution
        q = 1.0 / (1.0 + dist)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return q

class DCEC(nn.Module):
    def __init__(self, input_shape, filters=[32, 64, 128, 10], n_clusters=10, alpha=1.0):
        super(DCEC, self).__init__()
        
        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []
        
        # 计算特征维度
        self.cae = CAE(input_shape, filters).to(self.device)
        # 假设输入一个样本计算特征维度
        with torch.no_grad():
            dummy_input = torch.zeros((1,) + input_shape).to(self.device)
            features = self.cae.encoder(dummy_input)
            n_features = features.view(1, -1).size(1)
        
        self.clustering = ClusteringLayer(n_clusters, n_features).to(self.device)
        
    def forward(self, x):
        # 获取编码器特征
        features = self.cae.encoder(x)
        # 展平特征
        features_flat = features.view(features.size(0), -1)
        # 计算聚类分布
        q = self.clustering(features_flat)
        
        return q, features

    def pretrain(self, train_loader, epochs=200, lr=1e-4, optimizer=None, save_dir='results/temp'):
        """预训练CAE"""
        print('...Pretraining...')
        
        # 确保模型在GPU上
        self.cae = self.cae.to(self.device)
        
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.cae.parameters(),
                lr=lr,
                weight_decay=1e-4
            )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )
        
        self.cae.train()
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        # 添加CUDA事件来计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        for epoch in range(epochs):
            total_loss = 0
            start_event.record()
            
            for batch_idx, batch in enumerate(train_loader):
                if isinstance(batch, list):
                    x = batch[0]
                else:
                    x = batch
                
                # 将数据移至GPU
                x = x.to(self.device)
                
                # 数据增强
                noise = torch.randn_like(x, device=self.device) * 0.1
                x_noisy = x + noise
                x_noisy = torch.clamp(x_noisy, -1, 1)
                
                optimizer.zero_grad()
                x_recon = self.cae(x_noisy)
                
                mse_loss = F.mse_loss(x_recon, x)
                l1_loss = F.l1_loss(x_recon, x)
                loss = 0.8 * mse_loss + 0.2 * l1_loss
                
                total_loss += loss.item()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.cae.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 打印批次进度
                if (batch_idx + 1) % 20 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] '
                          f'Loss: {loss.item():.6f}')
            
            end_event.record()
            torch.cuda.synchronize()
            time_elapsed = start_event.elapsed_time(end_event) / 1000  # 转换为秒
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}, Time: {time_elapsed:.2f}s')
            
            scheduler.step()
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.cae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, f'{save_dir}/best_cae.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    def fit(self, train_loader, y=None, maxiter=2e4, tol=1e-3, update_interval=140, 
            optimizer=None, save_dir='./results/temp'):
        """训练DCEC模型
        Args:
            train_loader: PyTorch DataLoader, 训练数据加载器
            y: numpy array, 真实标签(如果有的话)
            maxiter: int, 最大迭代次数
            tol: float, 停止训练的阈值
            update_interval: int, 更新目标分布的间隔
            optimizer: torch.optim, 优化器
            save_dir: str, 模型保存路径
        """
        print('Update interval', update_interval)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 初始化优化器
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())

        # 确保模型在正确的设备上
        self.to(self.device)
        
        # Step 1: 预训练检查
        if not self.pretrained:
            print('...pretraining CAE using default parameters')
            self.pretrain(train_loader, save_dir=save_dir)

        # Step 2: 初始化聚类中心
        print('Initializing cluster centers with k-means.')
        features = []
        self.eval()
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(self.device)
                # 获取编码器特征
                feature = self.cae.encoder(x)
                feature = feature.view(feature.size(0), -1)
                features.append(feature.cpu())
        features = torch.cat(features, dim=0).numpy()
        
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(features)
        y_pred_last = np.copy(y_pred)
        self.clustering.weight.data = torch.tensor(kmeans.cluster_centers_).to(self.device)

        # Step 3: 深度聚类
        # 创建日志文件
        import csv
        logfile = open(os.path.join(save_dir, 'dcec_log.csv'), 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'loss'])
        logwriter.writeheader()

        self.train()
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                # 更新目标分布
                q_all = []
                self.eval()
                with torch.no_grad():
                    for batch_idx, (x, _) in enumerate(train_loader):
                        x = x.to(self.device)
                        q, _ = self(x)
                        q_all.append(q.cpu())
                q_all = torch.cat(q_all, dim=0)
                
                # 计算目标分布P
                p = self.target_distribution(q_all)
                
                # 评估聚类性能
                y_pred = torch.argmax(q_all, dim=1).numpy()
                if y is not None:
                    acc_score = np.round(metrics.acc(y, y_pred), 5)
                    nmi_score = np.round(metrics.nmi(y, y_pred), 5)
                    ari_score = np.round(metrics.ari(y, y_pred), 5)
                    print(f'Iter {ite}: ACC= {acc_score:.4f}, NMI= {nmi_score:.4f}, ARI= {ari_score:.4f}')
                    
                    # 记录日志
                    logwriter.writerow({
                        'iter': ite,
                        'acc': acc_score,
                        'nmi': nmi_score,
                        'ari': ari_score,
                        'loss': total_loss if 'total_loss' in locals() else 0
                    })

                # 检查停止条件
                delta_label = np.sum(y_pred != y_pred_last).astype(float) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print(f'delta_label {delta_label} < tol {tol}')
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # 训练一个周期
            total_loss = 0
            self.train()
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(self.device)
                optimizer.zero_grad()
                
                # 前向传播
                q, _ = self(x)
                
                # 获取当前批次的目标分布
                p_batch = p[batch_idx * train_loader.batch_size:
                           min((batch_idx + 1) * train_loader.batch_size, p.shape[0])]
                p_batch = p_batch.to(self.device)
                
                # 计算聚类损失(KL散度)
                kl_loss = F.kl_div(q.log(), p_batch, reduction='batchmean')
                # 计算重构损失
                recon_loss = F.mse_loss(x_recon, x)
                # 总损失
                loss = 0.1 * kl_loss + recon_loss  # 0.1是权重系数
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            # 定期保存模型
            if ite % (update_interval * 10) == 0:
                torch.save(self.state_dict(), 
                          os.path.join(save_dir, f'dcec_model_{ite}.pt'))

        # 保存最终模型
        logfile.close()
        torch.save(self.state_dict(), 
                  os.path.join(save_dir, 'dcec_model_final.pt'))

    @staticmethod
    def target_distribution(q):
        """计算目标分布P
        Args:
            q: 当前的软分配分布
        Returns:
            target: 目标分布
        """
        weight = q ** 2 / torch.sum(q, dim=0)
        return (weight.t() / torch.sum(weight, dim=1)).t()
