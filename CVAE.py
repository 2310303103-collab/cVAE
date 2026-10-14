import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

# cuda setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': False}

# 超参数配置
batch_size = 16       # 适当减小批次大小，防止显存不足
latent_size = 16       # 精简隐空间维度，防止过拟合
epochs = 800          # 适当增加训练轮次
learning_rate = 1e-3  # 调整学习率，确保训练稳定
beta_max = 0.012       # 降低 KL 散度的权重，优先保证几何参数重构精度 (MSE)
beta_warmup_epochs = 200  # KL 散度权重预热轮数


def plot_loss_curve(train_losses, test_losses):
    """绘制训练与测试 Loss 曲线图"""
    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=1.8)
    plt.plot(test_losses, label='Test Loss', color='#ff7f0e', linewidth=1.8, linestyle='--')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('CVAE Training & Validation Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()




# ==========================================
# 0. 自定义数据集
# ==========================================
class MetamaterialDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        df = pd.read_excel(file_path) # 请确保替换为你的实际文件路径
        
        # 剔除第7-10列（力学参数）有缺失值的行
        df = df.dropna(subset=df.columns[6:10])
        
        # 提取数据：特征x (第3-6列，几何参数)，条件c (第7-10列，力学参数)
        x_data = df.iloc[:, 2:6].values.astype(np.float32)
        c_data = df.iloc[:, 6:10].values.astype(np.float32)
        
        # 归一化 (Min-Max) 将数据缩放到 0-1 之间，这对于神经网络训练至关重要
        self.x_min, self.x_max = x_data.min(axis=0), x_data.max(axis=0)
        self.c_min, self.c_max = c_data.min(axis=0), c_data.max(axis=0)
        
        x_data = (x_data - self.x_min) / (self.x_max - self.x_min + 1e-8)
        c_data = (c_data - self.c_min) / (self.c_max - self.c_min + 1e-8)
        
        self.x = torch.tensor(x_data)
        self.c = torch.tensor(c_data)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.c[idx]

dataset = MetamaterialDataset('/Users/wangminda/Desktop/CVAE代码/result adjust.xlsx')

# 划分训练集和测试集 (80% 训练, 20% 测试)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)






# ==========================================
# 1. CVAE 模型定义
# ==========================================

class CVAE(nn.Module): # 定义条件变分自编码器 (CVAE) 的网络结构类，继承自 nn.Module
    def __init__(self, feature_size, latent_size, class_size): # 初始化方法，传入输入特征维度、隐空间维度和类别条件维度
        super(CVAE, self).__init__() # 调用父类 nn.Module 的初始化函数
        self.feature_size = feature_size # 保存输入特征的维度大小
        self.class_size = class_size # 保存类别条件的维度大小

        hidden_dim = 64

        # encoder：输入 (4 + 4 = 8维) -> 64 -> 32 -> 均值/方差 (latent_size)
        self.fc1  = nn.Linear(feature_size + class_size, hidden_dim) # 编码器第一层全连接层：输入为特征与条件拼接后的向量，输出 hidden_dim 维隐层特征
        self.fc_mid = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc21 = nn.Linear(hidden_dim // 2, latent_size) # 编码器输出层之一：将 hidden_dim 维特征映射到隐分布的均值向量 mu
        self.fc22 = nn.Linear(hidden_dim // 2, latent_size) # 编码器输出层之二：将 hidden_dim 维特征映射到隐分布的对数方差向量 logvar

        # decode：输入 (latent_size + 4) -> 32 -> 64 -> 输出 (4维)
        self.fc3 = nn.Linear(latent_size + class_size, hidden_dim//2) # 解码器第一层全连接层：输入为隐变量 z 与条件 c 拼接后的向量，输出 hidden_dim 维特征
        self.fc_mid_dec = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, feature_size) # 解码器输出层全连接层：将 hidden_dim 维特征映射回原始的特征维度大小

        self.elu = nn.ELU() # 定义指数线性单元 (ELU) 激活函数
        self.sigmoid = nn.Sigmoid() # 定义 Sigmoid 激活函数，将输出压缩到 0 到 1 之间

    def encode(self, x, c): # 定义编码前向传播函数，输入为图像特征 x 和条件 c，对应推断分布 Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # 在特征维度（维度 1）上将输入图像张量与条件张量进行拼接
        h1 = self.elu(self.fc1(inputs)) # 将拼接后的张量输入第一层全连接层，并经过 ELU 激活函数
        h2 = self.elu(self.fc_mid(h1)) # 增加了一个精简的中间过渡层
        z_mu = self.fc21(h2) # 计算隐分布的正态分布均值向量 mu
        z_var = self.fc22(h2) # 计算隐分布的正态分布对数方差向量 logvar
        return z_mu, z_var # 返回均值和对数方差

    def reparameterize(self, mu, logvar): # 定义重参数化技巧 (Reparameterization Trick) 函数
        std = torch.exp(0.5*logvar) # 通过对数方差计算标准差：std = exp(0.5 * logvar)
        eps = torch.randn_like(std) # 从标准正态分布中随机采样与标准差同形状的噪声张量 eps
        return mu + eps*std # 通过公式 z = mu + eps * std 计算并返回采样后的隐变量 z

    def decode(self, z, c): # 定义解码前向传播函数，输入为隐变量 z 和条件 c，对应生成分布 P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # 在特征维度（维度 1）上将隐变量 z 与条件张量 c 进行拼接
        h3 = self.elu(self.fc3(inputs)) # 将拼接后的张量输入解码器第一层全连接层，并经过 ELU 激活函数
        h4 = self.elu(self.fc_mid_dec(h3)) # 对应的解码中间层
        return self.sigmoid(self.fc4(h4)) # 通过最后一层全连接层和 Sigmoid 激活函数

    def forward(self, x, c): # 定义整个 CVAE 模型的前向传播全流程函数
        mu, logvar = self.encode(x, c) # 调用 encode 方法获取隐分布的均值 mu 和对数方差 logvar
        z = self.reparameterize(mu, logvar) # 调用重参数化函数，从分布中采样得到隐变量 z
        return self.decode(z, c), mu, logvar # 调用 decode 方法重构图像，并同时返回重构图像、均值 mu 和对数方差 logvar

# create a CVAE model
model = CVAE(feature_size=4, latent_size=latent_size, class_size=4).to(device) # 将 CVAE 模型实例化，特征维度为 4，隐空间维度为 latent_size，类别条件维度为 4，并将模型移动到 GPU 设备上
optimizer = optim.Adam(model.parameters(), lr=1e-4) # 使用 Adam 优化器优化模型参数，学习率设置为 1e-4





# ==========================================
# 2. CVAE 损失函数
# ==========================================

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, beta): # 定义 CVAE 的复合损失函数计算方法（重构损失 + KL 散度）
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # 计算隐分布与标准正态分布之间的 KL 散度正则化项
    return MSE + beta * KLD # 返回重构损失与 KL 散度的加和作为总损失





# ==========================================
# 3. 模拟训练过程与运行示例
# ==========================================

def train(epoch): # 定义单轮训练（Training Epoch）的函数
    model.train() # 将模型设置为训练模式
    train_loss = 0 # 初始化当前 epoch 的累计训练损失为 0

    current_beta = min(beta_max, beta_max * (epoch / beta_warmup_epochs))

    for batch_idx, (data, conditions) in enumerate(train_loader): # 遍历训练集数据加载器
        data, conditions = data.to(device), conditions.to(device)   # 将图像数据和条件张量移动到 GPU 设备上
        
        recon_batch, mu, logvar = model(data, conditions)   # 将数据和条件输入模型，前向传播获取输出
        optimizer.zero_grad()   # 清空上一步优化的梯度缓存
        loss = loss_function(recon_batch, data, mu, logvar,beta=current_beta) # 计算当前批次的 CVAE 总损失
        loss.backward() # 执行反向传播，计算各个参数的梯度
        train_loss += loss.item()   # 将当前批次损失累加到训练总损失中
        optimizer.step()    # 使用优化器根据梯度更新模型参数

        avg_train_loss = train_loss / len(train_loader.dataset)
    
    # 仅在第1个epoch或每50个epoch打印一次训练信息
    if epoch % 50 == 0 or epoch == 1:
        print('====> Epoch: {} Average train loss: {:.4f} | Current Beta: {:.6f}'.format(epoch, avg_train_loss, current_beta))
        
    return avg_train_loss # 返回当前 epoch 的平均损失用于画图



def test(epoch): # 定义模型测试与评估的函数
    model.eval() # 将模型设置为评估模式
    test_loss = 0 # 初始化测试集总损失为 0
    with torch.no_grad(): # 在测试阶段禁用梯度计算
        for i, (data, conditions) in enumerate(test_loader):    # 遍历测试集数据加载器
            data, conditions = data.to(device), conditions.to(device)   # 将图像数据和条件张量移动到 GPU 设备上
            recon_batch, mu, logvar = model(data, conditions)   # 将数据和条件输入模型，前向传播获取输出
            test_loss += loss_function(recon_batch, data, mu, logvar,beta=beta_max).item() # 累加当前批次的测试损失到总损失中

    avg_test_loss = test_loss / len(test_loader.dataset)   
    # 仅在第1个epoch或每50个epoch打印一次测试信息
    if epoch % 50 == 0 or epoch == 1:
        print('====> Epoch: {} Average test loss: {:.4f}'.format(epoch, avg_test_loss))
        
    return avg_test_loss


# 初始化用于绘制全局 Loss 曲线的列表
history_train_loss = []
history_test_loss = []


for epoch in range(1, epochs + 1): # 开始主循环，从第 1 轮训练到指定的 epochs 总轮数
    avg_train = train(epoch)    
    avg_test = test(epoch) 
    
    history_train_loss.append(avg_train)
    history_test_loss.append(avg_test)

    # 绘制 Loss 曲线图
plot_loss_curve(history_train_loss, history_test_loss)
    



# ==========================================
# 4. 逆向生成验证 (Inference / Sampling)
# ==========================================

# 训练完成后，丢弃 Encoder，只用 Decoder 进行“按需生成”
model.eval()
with torch.no_grad():
    print("\n--- 启动逆向生成测试 ---")
    # 1. 设定你期望的 4 维力学性能目标（直接输入真实的物理数值）
    raw_target_conditions = np.array([
        [-0.007243523,-0.007235683,127.21226,0.5316 ], 
        [-0.001389213,-0.001359494,40.63150609,0.7187 ],
        [-0.005635077,-0.005553868,39.39805037,0.7338 ]
    ], dtype=np.float32) # 生成 3 个满足该条件的候选几何构型

    # 2. 将输入的真实力学参数转换为归一化值，以便模型可以处理
    # 依赖于 dataset 实例中保存的 c_min 和 c_max
    norm_conditions = (raw_target_conditions - dataset.c_min) / (dataset.c_max - dataset.c_min + 1e-8)
    norm_conditions_tensor = torch.tensor(norm_conditions).to(device)

    num_samples = norm_conditions_tensor.size(0)

    # 3. 从标准正态分布 N(0, I) 中随机采样隐变量 Z
    random_z = torch.randn(num_samples, latent_size).to(device)

    # 4. 喂给 Decoder 产生符合要求的支架几何参数 (输出为 0-1 之间的归一化值)
    generated_norm_geometries = model.decode(random_z, norm_conditions_tensor)

    # 5. 反向归一化：将模型生成的 0-1 值还原为真实的几何物理尺寸
    # 依赖于 dataset 实例中保存的 x_min 和 x_max
    generated_norm_np = generated_norm_geometries.cpu().numpy()
    real_geometries = generated_norm_np * (dataset.x_max - dataset.x_min) + dataset.x_min

    print(f"输入的真实力学条件 C :\n{raw_target_conditions}")
    print(f"模型逆向生成的真实几何参数 X:\n{real_geometries}")




# ==========================================
# 5. 逆向生成验证与隐空间连续性抽查
# ==========================================
model.eval()
with torch.no_grad():
    print("\n--- 启动隐空间连续性与多解性抽查 ---")
    
    # 1. 设定【单一】你期望的 4 维力学性能目标
    target_condition = np.array([[-0.005635077,-0.005553868,39.39805037,0.7338]], dtype=np.float32) 
    
    # 2. 将输入参数归一化
    norm_condition = (target_condition - dataset.c_min) / (dataset.c_max - dataset.c_min + 1e-8)
    norm_condition_tensor = torch.tensor(norm_condition).to(device)

    # 3. 将这 1 个目标条件复制 5 份，准备生成 5 种不同的几何方案
    num_solutions = 5
    norm_conditions_expanded = norm_condition_tensor.repeat(num_solutions, 1)

    # 4. 从标准正态分布 N(0, I) 中随机采样 5 组【不同】的隐变量 Z
    random_z = torch.randn(num_solutions, latent_size).to(device)

    # 5. 喂给 Decoder，同一条件 + 不同噪声 = 不同的几何构型
    generated_norm_geometries = model.decode(random_z, norm_conditions_expanded)

    # 6. 反向归一化还原真实物理尺寸
    generated_norm_np = generated_norm_geometries.cpu().numpy()
    real_geometries = generated_norm_np * (dataset.x_max - dataset.x_min) + dataset.x_min

    print(f"设定的统一力学目标 C:\n{target_condition[0]}\n")
    print(f"模型基于该目标探索出的 {num_solutions} 种候选几何参数方案 X:")
    for i, geo in enumerate(real_geometries):
        print(f"方案 {i+1}: {geo}")



