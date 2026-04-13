import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard

from model import SNN_CNN_Hybrid
from dataset_robust import FlexibleBloodFlowDataset, sequence_sparse_collate


class DenseBlockManager:
    def __init__(self, x_seq_sparse_data, batch_size, spatial_shape=(100, 368)):
        self.x_seq_sparse_data = x_seq_sparse_data
        self.batch_size = batch_size
        self.spatial_shape = spatial_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_block_dense(self, block_idx, block_size):
        start_t = block_idx * block_size
        end_t = min(start_t + block_size, len(self.x_seq_sparse_data))
        actual_block_size = end_t - start_t

        dense_block = torch.zeros((actual_block_size, self.batch_size, 1, *self.spatial_shape), device=self.device)

        for i, t in enumerate(range(start_t, end_t)):
            b_coords, b_feats = self.x_seq_sparse_data[t]
            # 用保存的坐标和特征重建 SparseTensor
            sp_tensor = ME.SparseTensor(features=b_feats, coordinates=b_coords, device=self.device)
            # 转化为 GPU 稠密张量 [Batch, 1, H, W]
            dense_t, _ = sp_tensor.dense(shape=torch.Size([self.batch_size, 1, *self.spatial_shape]))
            dense_block[i] = dense_t

        return dense_block


def train_cross_env():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 1. 跨环境泛化配置
    # ==========================================
    TOTAL_STEPS = 5000  # 100ms / 20us = 5000
    BLOCK_SIZE = 100  # Checkpoint 显存块大小
    BATCH_SIZE = 2  # 单卡 48G 测试值
    ORIGINAL_SHAPE = (100, 368)

    # 训练环境配置
    train_env_config = {
        "/data/zm/2026.1.12_testdata/gaoyuzhi": 0.014611,
        "/data/zm/2026.1.12_testdata/1.15_150_680W": 0.0105,
    }

    # 验证环境配置 (网络从未见过的物理场景)
    val_env_config = {
        "/data/zm/2026.1.12_testdata/1.15_150_580W": 0.0114853,
    }

    # ==========================================
    # 2. 鲁棒数据集与 DataLoader
    # ==========================================
    train_ds = FlexibleBloodFlowDataset(train_env_config, T=1, seq_len=TOTAL_STEPS, dt_us=20)
    val_ds = FlexibleBloodFlowDataset(val_env_config, T=1, seq_len=TOTAL_STEPS, dt_us=20)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=sequence_sparse_collate,
                              num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=sequence_sparse_collate,
                            num_workers=4)

    # ==========================================
    # 3. 模型与优化器加载
    # ==========================================
    model = SNN_CNN_Hybrid(in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 初始化绘图记录列表与 TensorBoard Writer
    writer = SummaryWriter(log_dir='./runs/cross_env_experiment')
    train_loss_history = []
    val_loss_history = []

    # ==========================================
    # 4. 训练与跨环境验证循环
    # ==========================================
    epochs = 50
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss_sum = 0.0

        for batch_idx, (x_seq_sparse_data, y_true, d_values) in enumerate(train_loader):
            actual_batch_size = y_true.shape[0]
            y_true = y_true.to(device)
            d_values = d_values.to(device)

            optimizer.zero_grad()

            block_manager = DenseBlockManager(x_seq_sparse_data, actual_batch_size, spatial_shape=ORIGINAL_SHAPE)

            tau_c_pred = model(
                dataloader_or_generator=block_manager,
                total_steps=TOTAL_STEPS,
                block_size=BLOCK_SIZE,
                original_shape=ORIGINAL_SHAPE
            )

            d_values_expanded = d_values.view(-1, 1, 1, 1)
            v_pred = d_values_expanded / (tau_c_pred + 1e-8)

            y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)
            loss = F.mse_loss(v_pred, y_true_expanded)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            print(f"Epoch {epoch} [Train], Batch {batch_idx}: Loss = {loss.item():.4f}")

        # 计算当前 Epoch 的平均训练 Loss
        avg_train_loss = train_loss_sum / max(len(train_loader), 1)

        # --- 跨环境验证阶段 ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_idx, (x_seq_sparse_data, y_true, d_values) in enumerate(val_loader):
                actual_batch_size = y_true.shape[0]
                y_true = y_true.to(device)
                d_values = d_values.to(device)

                block_manager = DenseBlockManager(x_seq_sparse_data, actual_batch_size, spatial_shape=ORIGINAL_SHAPE)

                tau_c_pred = model(
                    dataloader_or_generator=block_manager,
                    total_steps=TOTAL_STEPS,
                    block_size=BLOCK_SIZE,
                    original_shape=ORIGINAL_SHAPE
                )

                d_values_expanded = d_values.view(-1, 1, 1, 1)
                v_pred = d_values_expanded / (tau_c_pred + 1e-8)

                y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)
                loss = F.mse_loss(v_pred, y_true_expanded)

                val_loss_sum += loss.item()

        # 计算当前 Epoch 的平均验证 Loss
        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        print(f"=== Epoch {epoch} Validation (Cross-Env): Avg Loss = {avg_val_loss:.4f} ===")

        # 记录到历史列表与 TensorBoard
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        writer.add_scalars('Loss', {'Train': avg_train_loss, 'Val': avg_val_loss}, epoch)

    # ==========================================
    # 5. 训练结束：关闭 Writer 并保存静态 Loss 曲线图
    # ==========================================
    writer.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(epochs), train_loss_history, label='Train Loss', color='blue', marker='o')
    plt.plot(range(epochs), val_loss_history, label='Validation (Cross-Env) Loss', color='red', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Training and Cross-Environment Validation Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    print("=> 训练结束，高精度 Loss 曲线已保存至当前目录下的 'loss_curve.png'，同时数据已同步至 TensorBoard。")


if __name__ == '__main__':
    train_cross_env()